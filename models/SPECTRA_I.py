import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from mamba_ssm import Mamba, Mamba2
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# Macro-Trend & Periodic Decoupling
class MTPD(nn.Module):
    def __init__(self, cutoff_freq=0.125):
        super(MTPD, self).__init__()
        # cutoff 
        self.cutoff_freq = cutoff_freq 

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        # FFT to frequency domain
        xf = torch.fft.rfft(x, dim=1)
        
        # low-pass
        mask = torch.zeros_like(xf)
        cutoff_idx = int(mask.shape[1] * self.cutoff_freq)
        mask[:, :cutoff_idx, :] = 1.0
        
        # trend & periodic components
        trend_periodic = torch.fft.irfft(xf * mask, n=L, dim=1)
        
        # high-frequency residual
        high_freq_residual = x - trend_periodic
        return trend_periodic, high_freq_residual

# Exogenous Context Synergizer
class ECS(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super(ECS, self).__init__()
        self.cross_attn1 = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, det_feat, res_feat, exo_feat):
        # Query: Endogenous Volatility, Trend Feature
        # Key, Value: Exogenous Context
        attn_out2, _ = self.cross_attn2(query=det_feat, key=exo_feat, value=exo_feat)
        attn_out1, _ = self.cross_attn1(query=res_feat, key=exo_feat, value=exo_feat)
        
        synergized_det = self.norm2(det_feat + self.dropout(attn_out2))
        synergized_res = self.norm1(res_feat + self.dropout(attn_out1))
        return synergized_det, synergized_res


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# Spectral-Temporal State-Space Engine & Block
class STSSE_Block(nn.Module):
    def __init__(self, configs):
        super(STSSE_Block, self).__init__()
        self.d_model = configs.d_model
        
        # frequency domain
        self.wavelet_high = Mamba(d_model=(configs.d_model + 7) // 2, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.wavelet_low = Mamba(d_model=(configs.d_model + 7) // 2, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        
        # Temporal domain
        self.time_forward = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.time_backward = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        
        self.norm_fwd = RMSNorm(configs.d_model)
        self.norm_bwd = RMSNorm(configs.d_model)
    
    def forward(self, x, x_low, x_high):
        w_low_out = self.wavelet_low(x_low)
        w_high_out = self.wavelet_high(x_high)
        
        t_fwd = self.time_forward(self.norm_fwd(x))
        t_bwd = self.time_backward(self.norm_bwd(x).flip(dims=[1])).flip(dims=[1])
        x_time = t_fwd + t_bwd + x
        
        return x_time, w_low_out, w_high_out


class STSSE_Engine(nn.Module):
    def __init__(self, configs, blocks, norm_layer=None):
        super(STSSE_Engine, self).__init__()
        self.WT = DWT1DForward(wave=configs.wavelet_type, J=1, mode='symmetric')
        self.IWT = DWT1DInverse(wave=configs.wavelet_type)
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

# Stochastic Boundary Estimator
class SBE(nn.Module):
    def __init__(self, d_model, pred_len, num_quantiles, num_channels):
        super(SBE, self).__init__()
        self.num_quantiles = num_quantiles
        self.pred_len = pred_len
        self.mid_idx = num_quantiles // 2
        
        self.proj_det = nn.Linear(d_model, pred_len)
        self.proj_stoch = nn.Linear(d_model * 2, pred_len * num_quantiles)
        
        self.scale = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, det_feat, stoch_feat):
        # det_feat : [B, Channels, d_model]
        # stoch_feat : [B, Channels, d_model]
        B, C, _ = det_feat.shape
        
        # deterministic
        det_out = self.proj_det(det_feat) # [B, C, L]
        det_out = det_out.permute(0, 2, 1).unsqueeze(1) # [B, 1, L, C] 
        
        # based on the deterministic feature and stochastic feature
        combined_feat = torch.cat([det_feat, stoch_feat], dim=-1) # [B, C, 2 * d_model]
        
        # stochastic bounds
        stoch_out = self.proj_stoch(combined_feat) # [B, C, L * Q]
        stoch_out = stoch_out.view(B, C, self.num_quantiles, self.pred_len)
        stoch_out = stoch_out.permute(0, 2, 3, 1) # [B, Q, L, C]
        
        # align quantiles: lower < mid < upper
        lower = -torch.abs(stoch_out[:, :self.mid_idx, :, :]) * self.scale
        # mid = torch.zeros_like(stoch_out[:, self.mid_idx:self.mid_idx+1, :, :])
        mid = stoch_out[:, self.mid_idx:self.mid_idx+1, :, :]
        upper = torch.abs(stoch_out[:, self.mid_idx+1:, :, :]) * self.scale
        
        stoch_bounds = torch.cat([lower, mid, upper], dim=1) # [B, Q, L, C]
        
        # Two-Stream Fusion
        final_out = det_out + stoch_bounds 
        return final_out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.num_quantiles = len(configs.quantiles)
        
        # normalization
        self.norm_layer = RevIN(configs.enc_in) if configs.use_norm and configs.norm_method == 'RevIN' else None
        
        # decomposition module
        self.decomposition = MTPD(cutoff_freq=configs.cutoff_freq)
        
        # embedding layers
        self.embed_det = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.embed_res = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        if True:
            self.embed_exo = nn.Linear(configs.seq_len + configs.pred_len, configs.d_model)
        else:
            self.embed_exo = nn.Linear(configs.seq_len, configs.d_model)
        
        # core
        self.STSSE = STSSE_Engine(
            configs,
            blocks=[STSSE_Block(configs) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        
        # exogenous context synergizer
        self.ECS = ECS(configs.d_model, num_heads=configs.cross_attn_heads)
        
        # stochastic boundary estimator
        self.SBE = SBE(configs.d_model, configs.pred_len, self.num_quantiles, configs.enc_in)
        
        # self.trend_exo_attn = nn.MultiheadAttention(configs.d_model, num_heads=4, batch_first=True)
        # self.trend_exo_norm = nn.LayerNorm(configs.d_model)
        
        # residual shortcut
        self.trend_shortcut = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x, x_mark, x_mark_h=None):
        # norm
        if self.norm_layer is not None:
            x = self.norm_layer(x, 'norm')
        elif self.configs.use_norm and self.configs.norm_method == 'NS':
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            
        B, L, D = x.shape
        
        # decomposition
        trend_periodic, high_freq_residual = self.decomposition(x)
        
        # trend_periodic: [B, L, D] -> permute -> [B, D, L] -> Linear -> [B, D, pred_len]
        base_trend = self.trend_shortcut(trend_periodic.permute(0, 2, 1))
        # [B, 1, pred_len, D]
        base_trend = base_trend.unsqueeze(1).permute(0, 1, 3, 2)
        
        # [B, L, D] -> [B, D, d_model]
        det_feat = self.embed_det(trend_periodic, None)
        res_feat = self.embed_res(high_freq_residual, None)
        
        # exogenous embedding
        if x_mark_h is not None:
            x_mark = torch.cat([x_mark, x_mark_h], dim=1) # [B, L + H, D_mark]
            
        exo_feat = self.embed_exo(x_mark.permute(0, 2, 1)) # [B, D_mark, d_model]
        
        # trend_attn_out, _ = self.trend_exo_attn(query=det_feat, key=exo_feat, value=exo_feat)

        # det_feat = self.trend_exo_norm(det_feat + trend_attn_out)

        
        synergized_det_feat, synergized_res_feat = self.ECS(det_feat, res_feat, exo_feat)
        
        det_out_feat = self.STSSE(synergized_det_feat)
        
        out = self.SBE(det_out_feat, synergized_res_feat) # [B, Q, L, D]
        
        out = out + base_trend
        
        # denorm
        if self.norm_layer is not None:
            out = self.norm_layer(out, 'denorm')
        elif self.configs.use_norm and self.configs.norm_method == 'NS':
            out = out * (stdev[:, 0, :].unsqueeze(1).unsqueeze(1))
            out = out + (means[:, 0, :].unsqueeze(1).unsqueeze(1))
            
        return out
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        return self.forecast(x_enc, x_mark_enc, x_mark_dec)