import numpy as np
import torch

def _check_shapes(pred, true):
    """
    适配 [B, Q, L, D] 和 [B, L, D] 输入。
    如果 pred 是 4 维而 true 是 3 维，默认取中位分位数 (Q//2) 进行点预测评估。
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    if pred.ndim == 4 and true.ndim == 3:
        # 假设第 1 维是分位数 Q，取中位数
        q_idx = pred.shape[1] // 2
        pred = pred[:, q_idx, :, :]
    
    return pred, true

def RSE(pred, true):
    pred, true = _check_shapes(pred, true)
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    pred, true = _check_shapes(pred, true)
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / (d + 1e-5)).mean(-1)

def MAE(pred, true):
    pred, true = _check_shapes(pred, true)
    return np.mean(np.abs(true - pred))

def MSE(pred, true):
    pred, true = _check_shapes(pred, true)
    return np.mean((true - pred) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    pred, true = _check_shapes(pred, true)
    return np.mean(np.abs((true - pred) / (true + 1e-5)))

def MSPE(pred, true):
    pred, true = _check_shapes(pred, true)
    return np.mean(np.square((true - pred) / (true + 1e-5)))

def PICP(pred, true):
    """
    Prediction Interval Coverage Probability
    计算真实值落在预测区间（通常由分位数 [low, high] 定义）内的比例。
    输入 pred: [B, Q, L, D], 其中 Q 至少包含两个分位数（如 0.05 和 0.95）
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    if pred.ndim != 4:
        return 0.0
    
    # 取最小和最大的分位数作为区间
    low = pred[:, 0, :, :]
    high = pred[:, -1, :, :]
    
    indicator = (true >= low) & (true <= high)
    return np.mean(indicator.astype(np.float32))

def ND(pred, true):
    """
    Normalized Deviation
    衡量预测值与真实值之间偏差
    """
    
    pred, true = _check_shapes(pred, true)
    return np.sum(np.abs(true - pred)) / (np.sum(np.abs(true)) + 1e-5)

def ROU(pred, true):
    """
    Quantile loss (normalized) for q=0.1 and q=0.9 and q=0.5.
    pred: [B, Q, L, D]
    true: [B, L, D]
    return: (rou10, rou90)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    if pred.ndim != 4 or true.ndim != 3:
        return 0.0, 0.0


    pred10 = pred[:, 0, :, :]
    pred50 = pred[:, pred.shape[1] // 2, :, :]
    pred90 = pred[:, -1, :, :]

    # Keep behavior aligned with other metrics: ignore padded/invalid zero labels.
    valid = (true != 0)
    denom = np.sum(np.abs(true[valid])) + 1e-5

    err10 = true[valid] - pred10[valid]
    err50 = true[valid] - pred50[valid]
    err90 = true[valid] - pred90[valid]

    ql10 = np.maximum((0.1 - 1.0) * err10, 0.1 * err10)
    ql50 = np.maximum((0.5 - 1.0) * err50, 0.5 * err50)
    ql90 = np.maximum((0.9 - 1.0) * err90, 0.9 * err90)

    rou10 = 2.0 * np.sum(ql10) / denom
    rou50 = 2.0 * np.sum(ql50) / denom
    rou90 = 2.0 * np.sum(ql90) / denom

    return rou10, rou50, rou90


def CRPS(pred, true):
    """
    Normalized Continuous Ranked Probability Score (NCRPS)
    pred: [B, Q, L, D]
    true: [B, L, D]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    if pred.ndim != 4 or true.ndim != 3:
        return 0.0

    q_num = pred.shape[1]
    if q_num <= 0:
        return 0.0

    # 若未显式传入分位点，采用常见默认：Q=3 -> [0.1, 0.5, 0.9]，其余均匀分布在 (0, 1)
    if q_num == 3:
        quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    else:
        quantiles = np.linspace(1.0 / (q_num + 1), q_num / (q_num + 1), q_num, dtype=np.float32)

    valid = (true != 0)
    denom = np.sum(np.abs(true[valid])) + 1e-5

    total_ql = 0.0
    for i, q in enumerate(quantiles):
        err = true[valid] - pred[:, i, :, :][valid]
        ql = np.maximum((q - 1.0) * err, q * err)
        total_ql += np.sum(ql)

    crps = 2.0 * total_ql / (q_num * denom)
    return crps
    

def metric(pred, true):
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    nd = ND(pred, true)
    picp = PICP(pred, true)
    rou10, rou50, rou90 = ROU(pred, true)
    crps = CRPS(pred, true)

    return mae, mse, rmse, nd, picp, rou10, rou50, rou90, crps