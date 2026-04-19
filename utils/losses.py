# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
    
    
class QuantileLoss(nn.Module):
    """
    preds  shape: [B, Q, L, V]  (批次, 分位数, 时间步, 变量数)
    target shape: [B, L, V]     (批次, 时间步, 变量数)
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = t.tensor(quantiles, dtype=t.float32)

    def forward(self, preds, target):
        # 安全校验
        assert not target.requires_grad
        B, Q, L, V = preds.shape
        B_t, L_t, V_t = target.shape
        assert B == B_t and L == L_t and V == V_t, "维度不匹配！"

        # 自动扩展target维度：[B,L,V] → [B,1,L,V] 匹配 preds
        target = target.unsqueeze(1)
        error = target - preds  # [B,Q,L,V]

        # 分位数广播计算 (Q → [1,Q,1,1])
        q = self.quantiles.view(1, -1, 1, 1).to(preds.device)
        
        # 分位数损失核心公式
        loss = t.where(error >= 0, q * error, (q - 1) * error)

        # 损失计算：先对分位数求和 → 全局平均
        loss = loss.sum(dim=1).mean()
        return loss
    

class SPECTRALoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], freq_weight=0.1, top_k=5):
        super().__init__()
        self.quantiles = t.tensor(quantiles, dtype=t.float32)
        self.freq_weight = freq_weight
        self.top_k = top_k

    def forward(self, preds, target):
        # preds/target shape: [B, Q, pred_len, D] 和 [B, pred_len, D]
        # 注意：这里的维度 D 在最后，需要跟代码对齐
        assert not target.requires_grad
        
        # 扩展维度 [B, pred_len, D] -> [B, 1, pred_len, D]
        target_exp = target.unsqueeze(1)
        error = target_exp - preds
        
        # --- 1. 时域：分位数损失 ---
        q = self.quantiles.view(1, -1, 1, 1).to(preds.device)
        quantile_loss = t.where(error >= 0, q * error, (q - 1) * error)
        loss_q = quantile_loss.sum(dim=1).mean()

        # --- 2. 频域损失 (基于预测长度动态开关) ---
        pred_len = preds.shape[2] # 获取当前预测长度
        loss_f = t.tensor(0.0, device=preds.device)
        
        # 核心逻辑：仅当预测长度 >= 24（至少包含一个日周期）时，才激活频域损失
        if pred_len >= 24:
            # 沿时间步维度进行 FFT
            fft_preds = t.fft.rfft(preds, dim=2, norm="forward")
            fft_target = t.fft.rfft(target_exp, dim=2, norm="forward")
            
            amp_preds = t.abs(fft_preds)
            amp_target = t.abs(fft_target)
            
            # 动态调整实际的 K 值 (防止极短序列时 top_k 越界)
            current_k = min(self.top_k, amp_target.shape[2])
            
            # 提取真实分布的最强频率索引
            _, topk_indices = t.topk(amp_target, k=current_k, dim=2)
            topk_indices_expand = topk_indices.expand(-1, preds.shape[1], -1, -1)
            
            topk_amp_preds = t.gather(amp_preds, 2, topk_indices_expand)
            topk_amp_target = t.gather(amp_target, 2, topk_indices)
            
            loss_f = F.l1_loss(topk_amp_preds, topk_amp_target.expand_as(topk_amp_preds))
        # print(f"Quantile Loss: {loss_q.item():.4f}, Freq Loss: {loss_f.item():.4f}")
        return loss_q + self.freq_weight * loss_f