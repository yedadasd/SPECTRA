import math
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from mamba_ssm import Mamba, Mamba2
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# Macro-Trend & Periodic Decoupling
class MTPD(nn.Module):
    def __init__(
        self,
        d_model=None,
        cutoff_freq=0.125,
        mode="fixed",
        temperature=0.03,
        context_dim=None,
        return_mask=False,):
        super(MTPD, self).__init__()

        assert mode in ["fixed", "attention"], \
            "mode must be either 'fixed' or 'attention'."

        self.cutoff_freq = cutoff_freq
        self.mode = mode
        self.temperature = temperature
        self.return_mask = return_mask

        if self.mode == "attention":
            if d_model is None:
                raise ValueError("d_model must be provided when mode='attention'.")

            # Frequency-domain attention.
            # Input:  [B, D, F]
            # Output: [B, D, F]
            self.freq_attention = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )

            # Optional exogenous-context conditioning.
            if context_dim is not None:
                self.context_proj = nn.Linear(context_dim, d_model)
            else:
                self.context_proj = None

            # Learnable balance between the original low-pass prior
            # and the adaptive attention mask.
            self.prior_logit = nn.Parameter(torch.tensor(1.0))

    def _fixed_lowpass_mask(self, xf):
        """
        Original hard low-pass mask.

        xf: [B, F, D]
        """
        B, F, D = xf.shape

        mask = torch.zeros_like(xf.real)
        cutoff_idx = int(F * self.cutoff_freq)

        # To avoid an empty low-frequency part when L is short.
        cutoff_idx = max(1, cutoff_idx)

        mask[:, :cutoff_idx, :] = 1.0
        return mask

    def _attention_lowpass_mask(self, xf, context=None):
        """
        Attention-guided soft low-pass mask.

        xf: [B, F, D]
        context: optional, [B, L, C] or [B, C]
        """
        B, F, D = xf.shape

        # Spectral energy: [B, F, D]
        spectral_energy = torch.log1p(torch.abs(xf))

        # Conv1d expects [B, D, F]
        spectral_energy = spectral_energy.permute(0, 2, 1)
        # print(spectral_energy.shape)
        # Adaptive spectral attention: [B, D, F]
        attn_logits = self.freq_attention(spectral_energy)

        # Optional context modulation.
        if context is not None and self.context_proj is not None:
            if context.dim() == 3:
                context_summary = context.mean(dim=1)  # [B, C]
            else:
                context_summary = context  # [B, C]

            context_bias = self.context_proj(context_summary).unsqueeze(-1)
            attn_logits = attn_logits + context_bias

        adaptive_mask = torch.sigmoid(attn_logits)  # [B, D, F]

        # Smooth version of the original low-frequency prior.
        freq_pos = torch.linspace(0, 1, F, device=xf.device).view(1, 1, F)
        low_freq_prior = torch.sigmoid(
            (self.cutoff_freq - freq_pos) / self.temperature
        )  # [1, 1, F]

        # Fuse original frequency prior and adaptive attention.
        prior_weight = torch.sigmoid(self.prior_logit)
        low_mask = prior_weight * low_freq_prior + \
            (1.0 - prior_weight) * adaptive_mask

        # Back to [B, F, D]
        low_mask = low_mask.permute(0, 2, 1)

        return low_mask

    def forward(self, x, context=None):
        """
        Args:
            x: [B, L, D]
            context: optional exogenous context, [B, L, C] or [B, C]

        Returns:
            trend_periodic: [B, L, D]
            high_freq_residual: [B, L, D]
        """
        B, L, D = x.shape

        # FFT to frequency domain.
        xf = torch.fft.rfft(x, dim=1)  # [B, F, D]

        if self.mode == "fixed":
            low_mask = self._fixed_lowpass_mask(xf)

        elif self.mode == "attention":
            low_mask = self._attention_lowpass_mask(xf, context=context)

        # Low-frequency trend-periodic component.
        trend_periodic = torch.fft.irfft(xf * low_mask, n=L, dim=1)

        # High-frequency residual.
        high_freq_residual = x - trend_periodic

        if self.return_mask:
            return trend_periodic, high_freq_residual, low_mask

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
        
        wave_len = pywt.Wavelet(configs.wavelet_type).dec_len
        d_model_wave = (configs.d_model + wave_len - 1) // 2
        
        # frequency domain
        self.wavelet_high = Mamba(d_model=d_model_wave, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.wavelet_low = Mamba(d_model=d_model_wave, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        
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
        
    def forward(self, det_feat, stoch_feat, return_components=False):
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

        if return_components:
            # det_out: [B, 1, L, C]  -- deterministic / regular component
            # stoch_bounds: [B, Q, L, C]  -- stochastic / residual component
            return final_out, det_out, stoch_bounds
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
        self.decomposition = MTPD(configs.enc_in, cutoff_freq=configs.cutoff_freq)
        
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


    def forecast(self, x, x_mark, x_mark_h=None, return_components=False):
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

        synergized_det_feat, synergized_res_feat = self.ECS(det_feat, res_feat, exo_feat)

        det_out_feat = self.STSSE(synergized_det_feat)

        if return_components:
            out, det_sbe, stoch_sbe = self.SBE(det_out_feat, synergized_res_feat,
                                               return_components=True)
            # Regular (deterministic) component: base_trend + SBE deterministic output
            regular = base_trend + det_sbe   # [B, 1, pred_len, D]
            # Residual (stochastic) component: SBE stochastic bounds
            residual = stoch_sbe             # [B, Q, pred_len, D]
        else:
            out = self.SBE(det_out_feat, synergized_res_feat) # [B, Q, L, D]

        out = out + base_trend

        # denorm
        if self.norm_layer is not None:
            out = self.norm_layer(out, 'denorm')
            if return_components:
                regular = self.norm_layer(regular, 'denorm')
                residual = self.norm_layer(residual, 'denorm')
        elif self.configs.use_norm and self.configs.norm_method == 'NS':
            out = out * (stdev[:, 0, :].unsqueeze(1).unsqueeze(1))
            out = out + (means[:, 0, :].unsqueeze(1).unsqueeze(1))
            if return_components:
                regular = regular * (stdev[:, 0, :].unsqueeze(1).unsqueeze(1))
                regular = regular + (means[:, 0, :].unsqueeze(1).unsqueeze(1))
                residual = residual * (stdev[:, 0, :].unsqueeze(1).unsqueeze(1))
                residual = residual + (means[:, 0, :].unsqueeze(1).unsqueeze(1))

        if return_components:
            return out, regular, residual
        return out
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_components=False):

        return self.forecast(x_enc, x_mark_enc, x_mark_dec,
                             return_components=return_components)