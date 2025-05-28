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
                    self.d_model = 1024
                    self.dropout = 0.05
            
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
            chunk_size = min(64, time_points // 50)  # 确保chunk_size不会太大
            print(f"Using chunk_size: {chunk_size}")
            self.model = Model(config, chunk_size=chunk_size).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            # 使用Huber Loss，对异常值更鲁棒
            criterion = nn.HuberLoss(delta=1.0)
            
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

    class EventAwareLoss(nn.Module):
        """事件感知损失函数"""
        def __init__(self, point_weight=0.7, event_weight=0.3, min_event_length=3):
            super().__init__()
            self.point_weight = point_weight
            self.event_weight = event_weight
            self.min_event_length = min_event_length
            self.mse = nn.MSELoss()
            
        def forward(self, pred, target):
            # 点级别损失
            point_loss = self.mse(pred, target)
            
            # 事件级别损失（基于连续异常段的一致性）
            event_loss = self.compute_event_consistency_loss(pred, target)
            
            return self.point_weight * point_loss + self.event_weight * event_loss
            
        def compute_event_consistency_loss(self, pred, target):
            """计算事件一致性损失"""
            # 计算重构误差
            reconstruction_error = torch.mean((pred - target) ** 2, dim=2)  # [batch, seq_len]
            
            # 使用滑动窗口计算局部一致性
            window_size = self.min_event_length
            consistency_loss = 0.0
            
            for i in range(reconstruction_error.shape[1] - window_size + 1):
                window_error = reconstruction_error[:, i:i+window_size]
                # 计算窗口内误差的方差，鼓励连续区域有相似的误差模式
                window_variance = torch.var(window_error, dim=1)
                consistency_loss += torch.mean(window_variance)
                
            return consistency_loss / (reconstruction_error.shape[1] - window_size + 1)

    class EnsembleMTSExample(BaseMethod):
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.models = []
            self.model_weights = []  # 存储模型权重
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lr = params.get('lr', 0.001)
            self.epochs = params.get('epochs', 10)
            self.batch_size = params.get('batch_size', 32)
            # 使用不同的chunk_size训练多个模型
            self.chunk_sizes = params.get('chunk_sizes', [32, 64, 128])
            # 事件检测参数
            self.event_detection_params = {
                'min_event_length': params.get('min_event_length', 3),
                'max_gap_length': params.get('max_gap_length', 2),
                'event_threshold_multiplier': params.get('event_threshold_multiplier', 1.2)
            }
            
        def train_valid_phase(self, tsData):
            print(f"Training data shape: {tsData.train.shape}")
            
            # 创建配置对象
            class Config:
                def __init__(self, seq_len, enc_in):
                    self.task_name = 'anomaly_detection'
                    self.seq_len = seq_len
                    self.enc_in = enc_in
                    self.d_model = 1024
                    self.dropout = 0.05
            
            # 获取训练数据的维度
            train_data = tsData.train
            # 正确解析数据维度：train_data的形状是(时间点数量, 特征数量)
            time_points, features = train_data.shape
            print(f"Time points: {time_points}, Features: {features}")
            
            # 为模型配置正确的参数
            config = Config(time_points, features)
            
            # 准备训练数据
            train_tensor = torch.FloatTensor(train_data).unsqueeze(0).to(self.device)  # [1, time_points, features]
            
            # 使用不同的chunk_size训练多个模型
            validation_scores = []  # 存储验证分数用于计算权重
            
            for i, chunk_size in enumerate(self.chunk_sizes):
                # 确保chunk_size不会太大
                actual_chunk_size = min(chunk_size, time_points // 10)
                print(f"Training model {i+1}/{len(self.chunk_sizes)} with chunk_size: {actual_chunk_size}")
                
                # 初始化模型
                model = Model(config, chunk_size=actual_chunk_size).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                # 使用事件感知损失函数
                criterion = EventAwareLoss(
                    point_weight=0.7, 
                    event_weight=0.3, 
                    min_event_length=self.event_detection_params['min_event_length']
                )
                
                # 训练循环
                model.train()
                best_loss = float('inf')
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    
                    # 前向传播
                    output = model.anomaly_detection(train_tensor)
                    loss = criterion(output, train_tensor)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    # 记录最佳损失用于权重计算
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                    
                    if (epoch+1) % 2 == 0 or epoch == 0:
                        print(f"Model {i+1}, Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")
                
                # 将训练好的模型添加到集合中
                self.models.append(model)
                validation_scores.append(1.0 / (1.0 + best_loss))  # 损失越小权重越大
            
            # 计算模型权重（基于验证性能）
            total_score = sum(validation_scores)
            self.model_weights = [score / total_score for score in validation_scores]
            print(f"Model weights: {[f'{w:.3f}' for w in self.model_weights]}")
            
            print(f"Ensemble training completed. Trained {len(self.models)} models with different chunk_sizes.")
            
        def post_process_for_events(self, scores, threshold):
            """事件级别后处理"""
            # 1. 基于阈值获得二值化结果
            binary_pred = (scores > threshold).astype(int)
            
            # 2. 移除短异常段
            min_event_length = self.event_detection_params['min_event_length']
            processed_pred = self.remove_short_events(binary_pred, min_event_length)
            
            # 3. 填充小间隔
            max_gap_length = self.event_detection_params['max_gap_length']
            processed_pred = self.fill_small_gaps(processed_pred, max_gap_length)
            
            return processed_pred
            
        def remove_short_events(self, binary_pred, min_length):
            """移除短异常事件"""
            result = binary_pred.copy()
            i = 0
            while i < len(result):
                if result[i] == 1:
                    # 找到异常段的结束位置
                    j = i
                    while j < len(result) and result[j] == 1:
                        j += 1
                    # 如果异常段长度小于最小长度，则移除
                    if j - i < min_length:
                        result[i:j] = 0
                    i = j
                else:
                    i += 1
            return result
            
        def fill_small_gaps(self, binary_pred, max_gap_length):
            """填充小间隔"""
            result = binary_pred.copy()
            i = 0
            while i < len(result) - 1:
                if result[i] == 1:
                    # 找到异常段的结束位置
                    j = i
                    while j < len(result) and result[j] == 1:
                        j += 1
                    # 检查后面是否有小间隔后的异常段
                    if j < len(result):
                        k = j
                        while k < len(result) and result[k] == 0:
                            k += 1
                        # 如果间隔长度小于最大间隔长度且后面还有异常段，则填充
                        if k < len(result) and result[k] == 1 and k - j <= max_gap_length:
                            result[j:k] = 1
                    i = j
                else:
                    i += 1
            return result
            
        def weighted_ensemble_predict(self, test_tensor):
            """加权集成预测"""
            all_scores = []
            
            # 对每个模型进行评估
            for i, model in enumerate(self.models):
                model.eval()
                with torch.no_grad():
                    # 获取重构结果
                    reconstructed = model.anomaly_detection(test_tensor)
                    
                    # 计算重构误差作为异常分数
                    reconstruction_error = torch.mean((test_tensor - reconstructed) ** 2, dim=2).squeeze(0)  # [time_points]
                    scores = reconstruction_error.cpu().numpy()
                    all_scores.append(scores)
                    print(f"Model {i+1} evaluation completed.")
            
            # 加权平均集成
            if len(self.model_weights) == len(all_scores):
                ensemble_scores = np.average(all_scores, axis=0, weights=self.model_weights)
                print(f"Weighted ensemble completed with weights: {[f'{w:.3f}' for w in self.model_weights]}")
            else:
                # 如果权重不匹配，使用简单平均
                ensemble_scores = np.mean(all_scores, axis=0)
                print("Simple average ensemble completed (weights not available)")
            
            return ensemble_scores

        def test_phase(self, tsData: MTSData):
            if not self.models:
                print("No models trained yet!")
                return
                
            # 获取测试数据
            test_data = tsData.test
            print(f"Test data shape: {test_data.shape}")
            
            # 准备测试数据
            test_tensor = torch.FloatTensor(test_data).unsqueeze(0).to(self.device)  # [1, time_points, features]
            
            # 使用加权集成预测
            ensemble_scores = self.weighted_ensemble_predict(test_tensor)
            
            # 应用事件级别后处理（可选）
            # 这里我们保留原始分数，后处理可以在评估阶段进行
            # 如果需要在这里应用后处理，可以取消下面的注释
            # threshold = np.percentile(ensemble_scores, 95)  # 使用95%分位数作为阈值
            # processed_binary = self.post_process_for_events(ensemble_scores, threshold)
            
            print("Ensemble prediction completed.")
            
            # 保存异常分数
            self.__anomaly_score = ensemble_scores
                
            return ensemble_scores
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            if not self.models:
                param_info = "Ensemble LightTS Model - Not trained yet"
            else:
                total_params = sum(sum(p.numel() for p in model.parameters()) for model in self.models)
                trainable_params = sum(sum(p.numel() for p in model.parameters() if p.requires_grad) for model in self.models)
                param_info = f"Ensemble LightTS Anomaly Detection Model\n"
                param_info += f"Number of models: {len(self.models)}\n"
                param_info += f"Chunk sizes: {self.chunk_sizes}\n"
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
    # method = "MTSExample"  # 单模型方法
    method = "EnsembleMTSExample"  # 集成模型方法
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "lr": 0.0005,  # 降低学习率以获得更稳定的训练
            "epochs": 50,  # 增加训练轮次
            "batch_size": 32,
            "chunk_sizes": [16, 32, 64, 128],  # 增加更多样的chunk_size组合
            # 事件检测参数
            "min_event_length": 3,  # 最小事件长度
            "max_gap_length": 2,    # 最大间隔长度
            "event_threshold_multiplier": 1.2  # 事件阈值倍数
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