from layers.TADNet_EncDec import *
import torch
import torch.nn as nn
import time 

class BasicsBlock(nn.Module):
    def __init__(self, cnn_num_inputs: int, num_channels: list, dropout,
                 static_cov_dim: list, hidden_size, quantiles_num,
                 num_heads, source_len, target_len, is_Regular=True, fourier_P=None) -> None:
        """
        fourier_P：当设置为傅里叶基底时, P为模型考虑的最大周期(必须取偶数)
        """
        super().__init__()

        self.fourier_P = fourier_P
        self.quantiles_num = quantiles_num

        self.cnn_block = TemporalConvNet(num_inputs=cnn_num_inputs, num_channels=num_channels, dropout=dropout)
        self.process_cov = ProcessStatic(static_cov_dim, hidden_size)
        self.grn = GRN(num_channels[-1], hidden_size)

        self.attention = AttentionNet(model_dim=hidden_size,
                                      num_heads=num_heads,
                                      source_len=source_len)

        if is_Regular:
            # Regular_Block
            if fourier_P is not None:
                # 采用傅里叶基
                self.last_layer = nn.Sequential(nn.Linear(hidden_size, fourier_P),
                                                SeasonalBasis(target_len, fourier_P))
            else:
                # 采用可学习基
                self.last_layer = nn.Linear(hidden_size, 1)
        else:
            self.last_layer = nn.Linear(hidden_size, quantiles_num)

    def forward(self, source_data, static_embedding, feature_future=None):
        """
        输出的长度为需要预测的长度
        """
        # (batch, feature, seq_len) -> (batch, num_channel, seq_len) -> (seq_len, batch, num_channel)
        cnn_output = self.cnn_block(source_data).permute(2, 0, 1)
        # (seq_len, batch, hidden_size)
        feature_past = self.grn(cnn_output, static_embedding)

        if feature_future is not None:

            # 规律项
            attn_input = torch.concat((feature_past, feature_future), dim=0)
            # (pred_len, batch, hidden_size)
            attn_output = self.attention(attn_input)
            if self.fourier_P is not None:
                
                output = self.last_layer(attn_output)
            else:
                # (pred_len, batch, w) -> (pred_len, batch, Q) -> (batch, Q, pred_len)
                output = self.last_layer(attn_output).permute(1, 2, 0)
        else:
            # 残余项
            # 噪声项的cnn_output已经包含了未来的信息（经由上一步预测所得）
            # (seq_len + pred_len, batch, hidden_size)
            attn_input = feature_past
            attn_output = self.attention(attn_input)
            # (batch, quantiles_num, pred_len)
            temp = self.last_layer(attn_output).permute(1, 2, 0)

            # 处理残余项输出的波动正负（波动下界为强制转负，波动上界转正）
            output_list = list()
            for i in range(self.quantiles_num):
                if i < self.quantiles_num // 2:
                    output_list.append(torch.unsqueeze(- torch.abs(temp[:, i, :]), dim=1))
                elif i == self.quantiles_num // 2:
                    output_list.append(torch.unsqueeze(temp[:, i, :], dim=1))
                else:
                    output_list.append(torch.unsqueeze(torch.abs(temp[:, i, :]), dim=1))
            output = torch.cat(output_list, dim=1)

        return output


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.regular = None
        self.remainder = None
        self.dropout = configs.dropout
        self.static_cov_dim = [configs.enc_in]
        self.hidden_size = configs.d_model
        self.quantiles_num = len(configs.quantiles)
        self.num_heads = configs.n_heads
        self.source_len = configs.seq_len
        self.target_len = configs.pred_len
        self.fourier_P = None
        self.num_time_cov = 4
        

        self.regular_block = BasicsBlock(cnn_num_inputs=5,
                                         num_channels=configs.num_channels,
                                         dropout=self.dropout,
                                         static_cov_dim=self.static_cov_dim,
                                         hidden_size=self.hidden_size,
                                         quantiles_num=self.quantiles_num,
                                         num_heads=self.num_heads,
                                         source_len=self.source_len,
                                         target_len=self.target_len,
                                         fourier_P=self.fourier_P)

        self.residual_block = BasicsBlock(cnn_num_inputs=5,
                                          num_channels=configs.num_channels,
                                          dropout=self.dropout,
                                          static_cov_dim=self.static_cov_dim,
                                          hidden_size=self.hidden_size,
                                          quantiles_num=self.quantiles_num,
                                          num_heads=self.num_heads,
                                          source_len=self.source_len,
                                          target_len=self.target_len,
                                          is_Regular=False
                                          )
        
        self.process_static = ProcessStatic(self.static_cov_dim, self.hidden_size)
        
        self.fc_cov_future = nn.Linear(self.num_time_cov, self.hidden_size)

    def forecast(self, source_data, time_cov_future, static_cov=None):
        """
        输出变量的维度(batch, feature, seq_len)
        """
        # 协变量处理
        # (batch, len(static_cov_dim), 1) -> (1, batch, embedding_size)
        static_embedding = self.process_static(static_cov)
        # (batch, num_time_cov, pred_len) -> (pred_len, batch, hidden_size)
        feature_future = self.fc_cov_future(time_cov_future.permute(2, 0, 1))
        # 规律项
        # (batch, 1, pred_len)
        regular = self.regular_block(source_data, static_embedding, feature_future)
        # 残余项
        # (bach, 1, pred_len) -> (batch, 1 + num_time_cov, pred_len)
        residual_input_future = torch.concat((regular, time_cov_future), dim=1)
        # (batch, 1 + num_time_cov, seq_len + pred_len)
        residual_input = torch.concat((source_data, residual_input_future), dim=-1)
        residual = self.residual_block(residual_input, static_embedding)

        # 输出
        output = regular + residual
        self.regular = regular
        self.remainder = residual 

        return output

    def split(self):
        return self.regular, self.remainder
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, D = x_enc.shape
        _, _, C = x_mark_enc.shape
        _, T, _ = x_mark_dec.shape
        
        # [B, L, D] -> [B, D, L]
        x_enc = x_enc.permute(0, 2, 1)
        # [B, L, C] -> [B, C, L]
        x_mark_enc = x_mark_enc.permute(0, 2, 1)
        # [B, T, C] -> [B, C, T]
        x_mark_dec = x_mark_dec.permute(0, 2, 1)
        # [B, C, L] -> [B, 1, C, L]
        x_mark_enc = x_mark_enc.unsqueeze(1)
        # [B, 1, C, L] -> [B, D, C, L]
        x_mark_enc = x_mark_enc.repeat(1, D, 1, 1)
        # [B, D, L] -> [B, D, 1, L]
        x_enc = x_enc.unsqueeze(2)
        # [B, D, 1+C, L]
        source_data = torch.concat((x_enc, x_mark_enc), dim=2)
        # [B, D, 1+C, L] -> [B*D, L, 1+C]
        source_data = source_data.permute(0, 1, 3, 2).reshape(B*D, L, (C + 1))
        
        # [B*D, 1+C, L]
        source_data = source_data.permute(0, 2, 1)
        
        # [B, C, T] -> [B, 1, C, T]
        x_mark_dec = x_mark_dec.unsqueeze(1)
        # [B, 1, C, T] -> [B, D, C, T]
        x_mark_dec = x_mark_dec.repeat(1, D, 1, 1)
        # [B, D, C, T] -> [B*D, T, C]
        x_mark_dec = x_mark_dec.permute(0, 1, 3, 2).reshape(B*D, T, C)
        
        # [B*D, C, T]
        time_cov_future = x_mark_dec.permute(0, 2, 1)
        
        # 根据维度D创建range再重复B次，得到维度为[B*D, 1, 1]的静态协变量
        static_cov = torch.arange(D).repeat(B).unsqueeze(1).unsqueeze(2).to(x_enc.device)
        
        # [B*D, Q, T]
        dec_out = self.forecast(source_data=source_data, time_cov_future=time_cov_future, static_cov=static_cov)
        # [B*D, Q, T] -> [B, D, Q, T] -> [B, Q, T, D]
        return dec_out.reshape(B, D, self.quantiles_num, T).permute(0, 2, 3, 1)