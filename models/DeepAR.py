import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.quantiles = configs.quantiles
        self.num_quantiles = len(self.quantiles)
        # 假设分位数列表的中间项是中位数 (例如 0.5)，推理时会把它作为下一步的输入
        self.mid_idx = self.num_quantiles // 2 

        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout if self.num_layers > 1 else 0.0

        self.rnn = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        
        # 直接映射到所有分位数
        self.projection = nn.Linear(self.hidden_size, self.c_out * self.num_quantiles)


    def forecast(self, x_enc_norm, means, stdev):
        bsz = x_enc_norm.shape[0]

        # 编码历史信息
        _, state = self.rnn(x_enc_norm)

        # 拿出历史的最后一步作为自回归的起点
        curr_input = x_enc_norm[:, -1:, :]
        pred_quantiles = []

        # 滚动预测
        for t in range(self.pred_len):
            out, state = self.rnn(curr_input, state)
            step_quantiles = self.projection(out) # [bsz, 1, c_out * num_quantiles]
            
            step_quantiles = step_quantiles.view(bsz, 1, self.num_quantiles, self.c_out)
            pred_quantiles.append(step_quantiles)

            # 提取预测的中位数 (mid_idx) 作为下一步的输入
            # 注意：这里 curr_input 必须保持在标准化尺度，因为网络是在标准化数据上训练的
            median_pred = step_quantiles[:, :, self.mid_idx, :] 
            curr_input = median_pred

        # 拼接所有时间步: [bsz, pred_len, num_quantiles, c_out]
        dec_out = torch.cat(pred_quantiles, dim=1) 

        # 反标准化
        dec_out = dec_out * stdev.unsqueeze(2) + means.unsqueeze(2)

        # 返回形状: [bsz, num_quantiles, pred_len, c_out]
        return dec_out.permute(0, 2, 1, 3)
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm = x_enc_norm / stdev

        if self.training:
            # --- 训练阶段：Teacher Forcing (使用真实标签作为下一步输入) ---
            x_dec_norm = (x_dec - means) / stdev

            # 编码历史信息
            _, state = self.rnn(x_enc_norm)

            # 构建 Decoder 的输入: [历史最后一步, 未来真实第1步, ..., 未来真实第N-1步]
            last_enc_step = x_enc_norm[:, -1:, :]
            future_gt = x_dec_norm[:, -self.pred_len:, :]
            
            if self.pred_len > 1:
                dec_input = torch.cat([last_enc_step, future_gt[:, :-1, :]], dim=1)
            else:
                dec_input = last_enc_step

            # 获得解码输出
            dec_hidden, _ = self.rnn(dec_input, state)
            dec_out = self.projection(dec_hidden) # [bsz, pred_len, c_out * num_quantiles]

            bsz, out_len, _ = dec_out.shape
            dec_out = dec_out.view(bsz, out_len, self.num_quantiles, self.c_out)

            # 反标准化
            dec_out = dec_out * stdev.unsqueeze(2) + means.unsqueeze(2)

            # [bsz, num_quantiles, pred_len, c_out]
            return dec_out.permute(0, 2, 1, 3) 
        else:
            # --- 推理阶段：自回归滚动预测 (Autoregressive Rollout) ---
            if self.task_name in ['long_term_forecast', 'short_term_forecast']:
                return self.forecast(x_enc_norm, means, stdev)
            return None