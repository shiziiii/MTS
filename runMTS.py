import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from EasyTSAD.Controller import TSADController

# ============= LightTS Model Implementation =============
class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super(IEBlock, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node
        self._build()

    def _build(self):
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )
        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)
        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2207.01186
    """
    def __init__(self, configs, chunk_size=24):
        """
        chunk_size: int, reshape T into [num_chunks, chunk_size]
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len if self.task_name in ['classification', 'anomaly_detection', 'imputation'] else configs.pred_len
        
        # Determine chunk_size and ensure complete division
        self.chunk_size = min(self.seq_len, chunk_size)
        if self.seq_len % self.chunk_size != 0:
            self.seq_len += (self.chunk_size - self.seq_len % self.chunk_size)
        self.num_chunks = self.seq_len // self.chunk_size

        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.dropout = configs.dropout
        
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)
            
        self._build()

    def _build(self):
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2,
            hid_dim=self.d_model // 2,
            output_dim=self.pred_len,
            num_node=self.enc_in
        )

        # Linear layer for each feature dimension from seq_len to pred_len
        self.ar = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):
        B, T, N = x.size()
        original_T = T  # Save original input length
        
        # Handle seq_len adjustment (padding or truncation)
        if T < self.seq_len:
            x = torch.cat([x, torch.zeros(B, self.seq_len - T, N, device=x.device)], dim=1)
        elif T > self.seq_len:
            x = x[:, :self.seq_len, :]
        T = self.seq_len

        # Apply AR to each feature dimension
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B * N, T)
        highway_reshaped = self.ar(x_reshaped)
        highway = highway_reshaped.view(B, N, self.pred_len).permute(0, 2, 1)

        # Continuous sampling
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N).permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        # Interval sampling
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N).permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        # Combine features and apply final layer
        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(B, N, -1).permute(0, 2, 1)
        out = self.layer_3(x3) + highway
        
        # Ensure output length matches original input length
        if out.size(1) > original_T:
            out = out[:, :original_T, :]
        elif out.size(1) < original_T:
            out = torch.cat([out, torch.zeros(B, original_T - out.size(1), N, device=out.device)], dim=1)
        
        return out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc, x_mark_enc):
        # Padding handled in encoder method
        enc_out = self.encoder(x_enc)
        # Output
        return self.projection(enc_out.reshape(enc_out.shape[0], -1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None

# ============= Main Execution =============
if __name__ == "__main__":
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    gctrl.set_dataset(
        dataset_type="MTS",  # Multivariate time series dataset type
        dirname="/home/ubuntu/qkWorkSpace/MTS/datasets",
        datasets=["machine-1", "machine-2", "machine-3"]
    )

    """============= Implement your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData

    class MTSExample(BaseMethod):
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.models = []
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lr = params.get('lr', 0.001)
            self.epochs = params.get('epochs', 10)
            self.chunk_sizes = params.get('chunk_sizes', [32, 64, 128])
            
            # 设置随机种子（如果提供）
            if 'seed' in params:
                seed = params.get('seed')
                print(f"设置随机种子: {seed}")
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
        def train_valid_phase(self, tsData):
            print(f"Training data shape: {tsData.train.shape}")
            
            # Get training data dimensions
            train_data = tsData.train
            time_points, features = train_data.shape
            print(f"Time points: {time_points}, Features: {features}")
            
            # Create config object for model parameters
            class ModelConfig:
                pass
                
            config = ModelConfig()
            config.task_name = 'anomaly_detection'
            config.seq_len = time_points
            config.enc_in = features
            config.d_model = 2048
            config.dropout = 0.4
            
            # Prepare training data
            train_tensor = torch.FloatTensor(train_data).unsqueeze(0).to(self.device)
            criterion = nn.HuberLoss(delta=1.0)
            
            # Train multiple models with different chunk_sizes
            for i, chunk_size in enumerate(self.chunk_sizes):
                actual_chunk_size = min(chunk_size, time_points // 10)
                print(f"Training model {i+1}/{len(self.chunk_sizes)} with chunk_size: {actual_chunk_size}")
                
                # Initialize model and optimizer
                model = Model(config, chunk_size=actual_chunk_size).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                
                # Training loop
                model.train()
                for epoch in range(self.epochs):
                    # Forward pass
                    output = model.anomaly_detection(train_tensor)
                    loss = criterion(output, train_tensor)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Print progress
                    if (epoch+1) % 2 == 0 or epoch == 0:
                        print(f"Model {i+1}, Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")
                
                self.models.append(model)
            
            print(f"Ensemble training completed. Trained {len(self.models)} models with different chunk_sizes.")
            
        def test_phase(self, tsData: MTSData):
            if not self.models:
                print("No models trained yet!")
                return
                
            # Prepare test data
            test_data = tsData.test
            print(f"Test data shape: {test_data.shape}")
            test_tensor = torch.FloatTensor(test_data).unsqueeze(0).to(self.device)
            
            # Evaluate each model and collect reconstruction errors
            all_scores = []
            for i, model in enumerate(self.models):
                model.eval()
                with torch.no_grad():
                    # Get reconstruction and calculate error
                    reconstructed = model.anomaly_detection(test_tensor)
                    reconstruction_error = torch.mean((test_tensor - reconstructed) ** 2, dim=2).squeeze(0)
                    all_scores.append(reconstruction_error.cpu().numpy())
                    print(f"Model {i+1}/{len(self.models)} evaluation completed.")
            
            # Ensemble predictions by averaging all model scores
            self.__anomaly_score = np.mean(all_scores, axis=0)
            print("Ensemble prediction completed.")
            return self.__anomaly_score
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            # Generate model statistics
            if not self.models:
                param_info = "Ensemble LightTS Model - Not trained yet"
            else:
                # Calculate parameter counts
                total_params = sum(sum(p.numel() for p in model.parameters()) for model in self.models)
                trainable_params = sum(sum(p.numel() for p in model.parameters() if p.requires_grad) for model in self.models)
                
                # Format statistics information
                param_info = (
                    f"Ensemble LightTS Anomaly Detection Model\n"
                    f"Number of models: {len(self.models)}\n"
                    f"Chunk sizes: {self.chunk_sizes}\n"
                    f"Total parameters: {total_params}\n"
                    f"Trainable parameters: {trainable_params}\n"
                    f"Learning rate: {self.lr}\n"
                    f"Training epochs: {self.epochs}\n"
                    f"Device: {self.device}\n"
                )
            
            # Save statistics to file
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    """============= Run your algo. ============="""
    training_schema = "mts"
    method = "MTSExample"  # Ensemble model method
    
    # Run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "lr": 0.001,
            "epochs": 100,
            "chunk_sizes": [64, 128, 256],  # Train multiple models with different chunk_sizes
            "seed": 22  # 为所有三个数据集设置相同的随机种子
        },
        preprocess="z-score"
    )
       
    """============= [EVALUATION SETTINGS] ============="""
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
        
    """============= [PLOTTING SETTINGS] ============="""
    # Plot anomaly scores for each curve
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )