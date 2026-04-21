import numpy as np
import torch
from src.model.enhanced_lstm import EnhancedLSTMModel as LSTMModel

MULTI_FACTOR_DIM = 224  # [自动修正] 多因子特征维度

def build_state(price_array, window_size, lstm_model_path, device='cpu', initial_cash=1e7):
    # 构造与MultiAssetTradingEnv一致的state（仅用于推理）
    asset_num = price_array.shape[1]
    lstm_model = LSTMModel(MULTI_FACTOR_DIM, 128, 3, MULTI_FACTOR_DIM, 0.3).to(device)  # [自动修正] 输入输出均为224
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.eval()
    t = price_array.shape[0]
    window = price_array[t-window_size:t]
    window = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-8)
    X = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = lstm_model(X).cpu().numpy().flatten()
    # 默认全现金、无持仓
    position = np.zeros(asset_num)
    cash = initial_cash
    prices = price_array[t-1]
    total_value = cash + np.sum(position * prices)
    position_ratio = (position * prices) / (total_value + 1e-8)
    cash_ratio = cash / (total_value + 1e-8)
    price_hist = price_array[t-window_size:t].flatten()
    state = np.concatenate([feats, position_ratio, [cash_ratio], price_hist])
    return state.astype(np.float32)

def optimize_portfolio(model, price_array, risk_aversion=1.0, method='rl', window_size=20, lstm_model_path='model_ckpt/best_lstm_multi_asset.pth', device='cpu'):
    '''
    根据RL模型和风险偏好生成最优权重。
    method: 'rl'（RL输出）、'equal'（等权）、'risk_aversion'（风险调整）
    '''
    asset_num = price_array.shape[1]
    if method == 'equal':
        return np.ones(asset_num) / asset_num
    elif method == 'rl':
        state = build_state(price_array, window_size, lstm_model_path, device)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            weights = model.actor(state_tensor).cpu().numpy().flatten()
        # 风险厌恶调整（softmax+风险系数）
        weights = np.exp(weights * risk_aversion)
        weights = weights / np.sum(weights)
        return weights
    else:
        raise ValueError('未知method') 