from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from EasyTSAD.Controller import TSADController
from lightTS import Model
import os

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # 设置所有machine-1的数据集路径
    
    
    gctrl.set_dataset(
        dataset_type="MTS",  # 多变量时间序列数据集类型
        dirname="/home/ubuntu/qkWorkSpace/MTS/datasets",  # 指向datasets目录
        datasets=["machine-1"]  # 列出所有子数据集
)

    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData

    class MTSExample(BaseMethod):
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lr = params.get('lr', 0.001)
            self.epochs = params.get('epochs', 10)
            self.batch_size = params.get('batch_size', 32)
            
        def train_valid_phase(self, tsData):
            print(f"Training data shape: {tsData.train.shape}")
            
            # 创建配置对象
            class Config:
                def __init__(self, seq_len, enc_in):
                    self.task_name = 'anomaly_detection'
                    self.seq_len = seq_len
                    self.enc_in = enc_in
                    self.d_model = 512
                    self.dropout = 0.1
            
            # 获取训练数据的维度
            train_data = tsData.train
            # 正确解析数据维度：train_data的形状是(时间点数量, 特征数量)
            time_points, features = train_data.shape
            print(f"Time points: {time_points}, Features: {features}")
            
            # 为模型配置正确的参数
            # 在lightTS模型中，seq_len应该是序列长度，enc_in应该是特征数量
            # 对于我们的数据，time_points是序列长度，features是特征数量
            config = Config(time_points, features)
            
            # 初始化模型
            # 使用较小的chunk_size，以减少内存使用
            chunk_size = min(24, time_points // 100)  # 确保chunk_size不会太大
            print(f"Using chunk_size: {chunk_size}")
            self.model = Model(config, chunk_size=chunk_size).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            
            # 准备训练数据 - 调整为模型期望的输入格式 [batch_size, seq_len, enc_in]
            # 我们的数据形状已经是(time_points, features)，与模型期望的(seq_len, enc_in)匹配
            # 只需添加batch维度
            train_tensor = torch.FloatTensor(train_data).unsqueeze(0).to(self.device)  # [1, time_points, features]
            
            # 训练循环
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                
                # 前向传播
                output = self.model.anomaly_detection(train_tensor)
                loss = criterion(output, train_tensor)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                if (epoch+1) % 2 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")
            
            print("Training completed.")
            
        def test_phase(self, tsData: MTSData):
            if self.model is None:
                print("Model not trained yet!")
                return
                
            # 获取测试数据
            test_data = tsData.test
            print(f"Test data shape: {test_data.shape}")
            
            # 准备测试数据 - 与训练阶段保持一致
            # 不需要转置，直接使用原始形状的数据
            test_tensor = torch.FloatTensor(test_data).unsqueeze(0).to(self.device)  # [1, time_points, features]
            
            # 模型评估
            self.model.eval()
            with torch.no_grad():
                # 获取重构结果
                reconstructed = self.model.anomaly_detection(test_tensor)
                
                # 计算重构误差作为异常分数
                # 在特征维度上计算均方误差
                reconstruction_error = torch.mean((test_tensor - reconstructed) ** 2, dim=2).squeeze(0)  # [time_points]
                scores = reconstruction_error.cpu().numpy()
                
                # 保存异常分数
                self.__anomaly_score = scores
                
            return scores
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            if self.model is None:
                param_info = "LightTS Model - Not trained yet"
            else:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                param_info = f"LightTS Anomaly Detection Model\n"
                param_info += f"Total parameters: {total_params}\n"
                param_info += f"Trainable parameters: {trainable_params}\n"
                param_info += f"Learning rate: {self.lr}\n"
                param_info += f"Training epochs: {self.epochs}\n"
                param_info += f"Batch size: {self.batch_size}\n"
                param_info += f"Device: {self.device}\n"
            
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    """============= Run your algo. ============="""
    # Specifying methods and training schemas

    training_schema = "mts"
    method = "MTSExample"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "lr": 0.001,
            "epochs": 10,
            "batch_size": 32,
        },
        # use which method to preprocess original data. 
        # Default: raw
        # Option: 
        #   - z-score(Standardlization), 
        #   - min-max(Normalization), 
        #   - raw (original curves)
        preprocess="z-score", 
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
    
    # plot anomaly scores for each curve
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )