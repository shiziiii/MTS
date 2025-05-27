import torch
import torch.nn as nn
import torch.nn.functional as F


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

        x = x.permute(0, 2, 1)

        return x


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
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        if configs.task_name == 'long_term_forecast' or configs.task_name == 'short_term_forecast':
            self.chunk_size = min(configs.pred_len, configs.seq_len, chunk_size)
        else:
            self.chunk_size = min(configs.seq_len, chunk_size)
        # assert (self.seq_len % self.chunk_size == 0)
        if self.seq_len % self.chunk_size != 0:
            self.seq_len += (self.chunk_size - self.seq_len % self.chunk_size)  # padding in order to ensure complete division
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

        # 修改self.ar层，使其能够处理MTS数据格式
        # 对于每个特征维度，创建一个从seq_len到pred_len的线性层
        self.ar = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):
        B, T, N = x.size()
        original_T = T  # 保存原始输入长度
        
        # 处理seq_len调整的情况
        # 如果模型的seq_len被调整过（由于chunk_size整除要求），我们需要对输入进行padding
        if T != self.seq_len:
            if T < self.seq_len:
                # 如果输入序列长度小于模型期望的seq_len，进行padding
                padding_size = self.seq_len - T
                x = torch.cat([x, torch.zeros(B, padding_size, N, device=x.device)], dim=1)
                T = self.seq_len
            else:
                # 如果输入序列长度大于模型期望的seq_len，进行截断
                x = x[:, :self.seq_len, :]
                T = self.seq_len

        # 修改highway计算，确保维度匹配
        # x的形状是(B, T, N)，我们需要对每个特征维度分别应用AR
        # 将x重塑为(B*N, T)，然后应用AR，最后重塑回(B, pred_len, N)
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B * N, T)  # (B*N, T)
        highway_reshaped = self.ar(x_reshaped)  # (B*N, pred_len)
        highway = highway_reshaped.view(B, N, self.pred_len).permute(0, 2, 1)  # (B, pred_len, N)

        # continuous sampling
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        # interval sampling
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = torch.cat([x1, x2], dim=-1)

        x3 = x3.reshape(B, N, -1)
        x3 = x3.permute(0, 2, 1)

        out = self.layer_3(x3)

        out = out + highway
        
        # 确保输出长度与原始输入长度一致
        if out.size(1) != original_T:
            if out.size(1) > original_T:
                # 如果输出长度大于原始输入长度，进行截断
                out = out[:, :original_T, :]
            else:
                # 如果输出长度小于原始输入长度，进行padding
                padding_size = original_T - out.size(1)
                out = torch.cat([out, torch.zeros(B, padding_size, N, device=out.device)], dim=1)
        
        return out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc, x_mark_enc):
        # padding
        x_enc = torch.cat([x_enc, torch.zeros((x_enc.shape[0], self.seq_len-x_enc.shape[1], x_enc.shape[2])).to(x_enc.device)], dim=1)

        enc_out = self.encoder(x_enc)

        # Output
        output = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None