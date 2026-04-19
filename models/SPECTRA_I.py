import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from mamba_ssm import Mamba
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# ---------------------------------------------------------
# 1. MTPD: Macro-Trend & Periodic Decoupling
# 创新点：不再使用普通的 Attention，而是使用基于频域 (FFT) 的低通滤波器
# 显式地将电力数据分为“主导周期（低频基座）”和“高频随机扰动（残差）”
# ---------------------------------------------------------
class MTPD(nn.Module):
    def __init__(self, cutoff_freq=0.125):
        super(MTPD, self).__init__()
        self.cutoff_freq = cutoff_freq  # 截断频率比例，控制周期基座的平滑度

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        # FFT 变换到频域
        xf = torch.fft.rfft(x, dim=1)
        
        # 构建理想低通滤波器掩码
        mask = torch.zeros_like(xf)
        cutoff_idx = int(mask.shape[1] * self.cutoff_freq)
        mask[:, :cutoff_idx, :] = 1.0
        
        # 滤波并逆变换回时域作为确定性的基准周期 (Macro-Trend & Periodic)
        trend_periodic = torch.fft.irfft(xf * mask, n=L, dim=1)
        
        # 高频部分作为随机性残差 (High-Frequency Volatility)
        high_freq_residual = x - trend_periodic
        return trend_periodic, high_freq_residual

# ---------------------------------------------------------
# 2. ECS: Exogenous Context Synergizer
# 创新点：不与全局特征混合，而是将外生变量（天气/市场）定向路由给“高频残差”
# 物理意义：外生突发事件直接驱动系统的高频不确定性波动
# ---------------------------------------------------------
class ECS(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super(ECS, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, res_feat, exo_feat):
        # Query: 高频残差特征 (Endogenous Volatility)
        # Key, Value: 外生变量特征 (Exogenous Context)
        attn_out, _ = self.cross_attn(query=res_feat, key=exo_feat, value=exo_feat)
        # 残差连接与归一化：将外部冲击吸收到内生高频波动中
        synergized_res = self.norm(res_feat + self.dropout(attn_out))
        return synergized_res

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# ---------------------------------------------------------
# 3. STSSE: Spectral-Temporal State-Space Engine & Block
# 重命名并重构原 KarmaBlock。专注于处理确定性的周期基座数据。
# ---------------------------------------------------------
class STSSE_Block(nn.Module):
    def __init__(self, configs):
        super(STSSE_Block, self).__init__()
        self.d_model = configs.d_model
        
        # 频域/波段状态空间建模
        self.wavelet_high = Mamba(d_model=(configs.d_model + 7) // 2, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.wavelet_low = Mamba(d_model=(configs.d_model + 7) // 2, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        
        # 时域状态空间建模
        self.time_forward = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.time_backward = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        
        self.norm_fwd = RMSNorm(configs.d_model)
        self.norm_bwd = RMSNorm(configs.d_model)
    
    def forward(self, x, x_low, x_high):
        w_low_out = self.wavelet_low(x_low)
        w_high_out = self.wavelet_high(x_high)
        
        # 双向 Mamba 处理时域特征
        t_fwd = self.time_forward(self.norm_fwd(x))
        t_bwd = self.time_backward(self.norm_bwd(x).flip(dims=[1])).flip(dims=[1])
        x_time = t_fwd + t_bwd + x
        
        return x_time, w_low_out, w_high_out

class STSSE_Engine(nn.Module):
    def __init__(self, configs, blocks, norm_layer=None):
        super(STSSE_Engine, self).__init__()
        self.WT = DWT1DForward(wave='db4', J=1, mode='symmetric')
        self.IWT = DWT1DInverse(wave='db4')
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer
    
    def forward(self, x):
        yl, yhs = self.WT(x)
        xl, xhs = yl, yhs[0]
        for block in self.blocks:
            x, xl, xhs = block(x, xl, xhs)
        x_out = self.IWT((xl, [xhs])) + x
        if self.norm is not None:
            x_out = self.norm(x_out)
        return x_out

# ---------------------------------------------------------
# 4. SBE: Stochastic Boundary Estimator
# 创新点：明确的双流融合机制。将确定性基座与外生驱动的不确定性残差相加
# ---------------------------------------------------------
class SBE(nn.Module):
    def __init__(self, d_model, pred_len, num_quantiles, num_channels):
        super(SBE, self).__init__()
        self.num_quantiles = num_quantiles
        self.pred_len = pred_len
        self.mid_idx = num_quantiles // 2
        
        # 独立映射头
        self.proj_det = nn.Linear(d_model, pred_len)
        self.proj_stoch = nn.Linear(d_model * 2, pred_len * num_quantiles)
        
        self.scale = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, det_feat, stoch_feat):
        # det_feat (STSSE 输出): [B, Channels, d_model]
        # stoch_feat (ECS 输出): [B, Channels, d_model]
        B, C, _ = det_feat.shape
        
        # 1. 计算确定性预测基座
        det_out = self.proj_det(det_feat) # [B, C, L]
        det_out = det_out.permute(0, 2, 1).unsqueeze(1) # [B, 1, L, C] -> 适配 Q 维度
        
        combined_feat = torch.cat([det_feat, stoch_feat], dim=-1) # [B, C, 2 * d_model]
        
        # 2. 计算多维分位数残差波动
        stoch_out = self.proj_stoch(combined_feat) # [B, C, L * Q]
        stoch_out = stoch_out.view(B, C, self.num_quantiles, self.pred_len)
        stoch_out = stoch_out.permute(0, 2, 3, 1) # [B, Q, L, C]
        
        # 3. 强制不对称性边界 (保证分位数不交叉的物理约束)
        lower = -torch.abs(stoch_out[:, :self.mid_idx, :, :]) * self.scale
        # mid = torch.zeros_like(stoch_out[:, self.mid_idx:self.mid_idx+1, :, :])
        mid = stoch_out[:, self.mid_idx:self.mid_idx+1, :, :]
        upper = torch.abs(stoch_out[:, self.mid_idx+1:, :, :]) * self.scale
        
        stoch_bounds = torch.cat([lower, mid, upper], dim=1) # [B, Q, L, C]
        
        # 4. 双流融合：基准点位 + 动态概率边界
        final_out = det_out + stoch_bounds 
        return final_out

# ---------------------------------------------------------
# 5. SPECTRA 最终模型
# ---------------------------------------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.num_quantiles = len(configs.quantiles)
        
        # 数据归一化
        self.norm_layer = RevIN(configs.enc_in) if configs.use_norm and configs.norm_method == 'RevIN' else None
        
        # 核心模块 1: MTPD 分解
        self.decomposition = MTPD(cutoff_freq=0.25)
        
        # 倒置嵌入层 (Channel Independence + Inverted Embedding)
        self.embed_det = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.embed_res = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        if True:
            self.embed_exo = nn.Linear(configs.seq_len + configs.pred_len, configs.d_model)
        else:
            self.embed_exo = nn.Linear(configs.seq_len, configs.d_model) # 为外生 covariates 提供特征嵌入
        
        # 核心模块 2: STSSE 引擎 (处理确定性基座)
        self.STSSE = STSSE_Engine(
            configs,
            blocks=[STSSE_Block(configs) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        
        # 核心模块 3: ECS 外生路由器 (融合天气/市场因素至残差)
        self.ECS = ECS(configs.d_model, num_heads=4)
        
        # 核心模块 4: SBE 双流随机边界估计器
        self.SBE = SBE(configs.d_model, configs.pred_len, self.num_quantiles, configs.enc_in)
        
        self.trend_exo_attn = nn.MultiheadAttention(configs.d_model, num_heads=4, batch_first=True)
        self.trend_exo_norm = nn.LayerNorm(configs.d_model)
        
        self.trend_shortcut = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x, x_mark, x_mark_h=None):
        # 1. 归一化
        if self.norm_layer is not None:
            x = self.norm_layer(x, 'norm')
        elif self.configs.use_norm and self.configs.norm_method == 'NS':
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            
        B, L, D = x.shape
        
        # 2. MTPD: 宏观趋势与高频残差解耦
        trend_periodic, high_freq_residual = self.decomposition(x)
        
        # trend_periodic: [B, L, D] -> permute -> [B, D, L] -> Linear -> [B, D, pred_len]
        base_trend = self.trend_shortcut(trend_periodic.permute(0, 2, 1))
        # 升维适配 SBE 输入: [B, 1, pred_len, D]
        base_trend = base_trend.unsqueeze(1).permute(0, 1, 3, 2)
        
        # 3. 特征倒置嵌入 (Inverted Representation)
        # [B, L, D] -> [B, D, d_model]
        det_feat = self.embed_det(trend_periodic, None)
        res_feat = self.embed_res(high_freq_residual, None)
        
        # 外生变量编码 (将 L 维的时间步特征化为 d_model 维度)
        # x_mark 维度通常为 [B, L, D_mark], 我们将其映射并通过 channel 转换
        if x_mark_h is not None:
            x_mark = torch.cat([x_mark, x_mark_h], dim=1) # [B, L + H, D_mark]
            
        exo_feat = self.embed_exo(x_mark.permute(0, 2, 1)) # [B, D_mark, d_model]
        
        trend_attn_out, _ = self.trend_exo_attn(query=det_feat, key=exo_feat, value=exo_feat)

        # 残差连接并归一化
        det_feat = self.trend_exo_norm(det_feat + trend_attn_out)
        
        # 4. STSSE 引擎：计算确定性基准特征
        det_out_feat = self.STSSE(det_feat)
        
        # 5. ECS 模块：外生变量跨模态路由至高频残差
        # 让外生变量驱动内部的高频不确定性
        synergized_res_feat = self.ECS(res_feat, exo_feat)
        
        # 6. SBE: 双流随机边界估计，输出分位数预测
        out = self.SBE(det_out_feat, synergized_res_feat) # [B, Q, L, D]
        
        out = out + base_trend
        
        # 7. 反归一化
        if self.norm_layer is not None:
            out = self.norm_layer(out, 'denorm')
        elif self.configs.use_norm and self.configs.norm_method == 'NS':
            out = out * (stdev[:, 0, :].unsqueeze(1).unsqueeze(1))
            out = out + (means[:, 0, :].unsqueeze(1).unsqueeze(1))
            
        return out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_mark_dec)