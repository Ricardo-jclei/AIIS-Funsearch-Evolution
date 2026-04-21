import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from src.model.enhanced_lstm import EnhancedLSTMModel
import yaml
import os

class MultiAssetTradingEnv(gym.Env):
    def __init__(self, price_array, feature_array, window_size=20, initial_cash=1e7, device='cpu', lstm_model_path='model_ckpt/best_lstm_multi_asset.pth', lstm_input_size=5, asset_num=5, cost_rate=None, sharpe_window=20, reward_type='sharpe', fee_rate=None, slippage_rate=None, enable_slippage=None, include_lstm_features=True):
        # 读取配置文件
        config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        # 优先使用传参，否则用配置文件，否则默认
        self.fee_rate = fee_rate if fee_rate is not None else config.get('fee_rate', 0.001)
        self.slippage_rate = slippage_rate if slippage_rate is not None else config.get('slippage_rate', 0.0)
        self.enable_slippage = enable_slippage if enable_slippage is not None else config.get('enable_slippage', False)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.asset_num = asset_num
        self.t = self.window_size
        self.cash = self.initial_cash
        self.position = np.zeros(self.asset_num)
        self.portfolio_value = self.initial_cash
        super().__init__()
        self.price_array = price_array  # shape: (T, asset_num)
        self.feature_array = feature_array  # shape: (T, N*feature_dim)
        self.device = device
        self.cost_rate = self.fee_rate  # 兼容旧参数
        self.lstm_input_size = lstm_input_size
        self.include_lstm_features = include_lstm_features
        if self.include_lstm_features:
            self.lstm_model = EnhancedLSTMModel(self.lstm_input_size, 128, 3, self.lstm_input_size, 0.3).to(device)
            self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
            self.lstm_model.eval()
        # 状态空间shape自动推断，确保与实际state一致
        dummy_state = self._get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=dummy_state.shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(asset_num,), dtype=np.float32)
        self.reset()
        self.sharpe_window = sharpe_window
        self.returns_window = []
        self.reward_type = reward_type  # 新增reward_type

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.t = self.window_size
        self.cash = self.initial_cash
        self.position = np.zeros(self.asset_num)  # 每只资产持仓股数
        self.portfolio_value = self.initial_cash
        self.returns_window = []  # 重置收益窗口
        print(f"[Reset] Initial position={self.position}, cash={self.cash}, price={self.price_array[self.t]}")
        return self._get_state(), {}

    def step(self, action):
        # 动作为资金分配权重，和为1，目标持仓按权重分配，不允许卖空
        action = np.clip(action, 0, 1)
        # 动作空间硬约束：单一资产最大权重0.7
        action = np.minimum(action, 0.7)
        action = action / (np.sum(action) + 1e-8)  # 再归一化，防止数值漂移
        prices = self.price_array[self.t]  # 当前价格 shape: (N,)
        # 滑点处理
        if self.enable_slippage:
            buy_prices = prices * (1 + self.slippage_rate)
            sell_prices = prices * (1 - self.slippage_rate)
        else:
            buy_prices = prices
            sell_prices = prices
        total_value = self.cash + np.sum(self.position * prices)
        # 目标持仓股数
        target_value = total_value * action
        target_shares = (target_value // prices).astype(int)
        # 实际买卖股数
        delta_shares = target_shares - self.position
        # 买入
        buy_shares = np.where(delta_shares > 0, delta_shares, 0)
        buy_cost = np.sum(buy_shares * buy_prices)
        # 卖出
        sell_shares = np.where(delta_shares < 0, -delta_shares, 0)
        sell_gain = np.sum(sell_shares * sell_prices)
        # 手续费
        fee = (buy_cost + sell_gain) * self.fee_rate
        # 现金和持仓变动
        self.position += delta_shares
        self.cash += sell_gain - buy_cost - fee
        self.cash = np.clip(self.cash, 0, 1e12)
        self.position = np.clip(self.position, 0, 1e8)
        # 计算reward
        prev_value = self.portfolio_value
        self.portfolio_value = self.cash + np.sum(self.position * prices)
        # --- reward优化 ---
        done = False
        # 计算持仓比例
        position_ratio = (self.position * prices) / (self.portfolio_value + 1e-8)
        if self.portfolio_value < 1e4:
            reward = -2.0
            done = True
        else:
            # 日收益
            ret = (self.portfolio_value - prev_value) / (prev_value + 1e-8)
            self.returns_window.append(ret)
            if len(self.returns_window) > self.sharpe_window:
                self.returns_window = self.returns_window[-self.sharpe_window:]
            # 计算reward_type
            reward = 0.0
            if len(self.returns_window) >= 2:
                mean_ret = np.mean(self.returns_window)
                std_ret = np.std(self.returns_window) + 1e-8
                downside_returns = [r for r in self.returns_window if r < 0]
                sortino = mean_ret / (np.std(downside_returns) + 1e-8) if len(downside_returns) > 0 else mean_ret / (std_ret)
                cum_returns = np.cumprod([1 + r for r in self.returns_window])
                peak = np.maximum.accumulate(cum_returns)
                max_drawdown = np.max((peak - cum_returns) / (peak + 1e-8))
                calmar = mean_ret / (max_drawdown + 1e-8)
                volatility = std_ret
                if self.reward_type == 'sharpe':
                    reward = mean_ret / std_ret
                elif self.reward_type == 'sortino':
                    reward = sortino
                elif self.reward_type == 'calmar':
                    reward = calmar
                elif self.reward_type == 'sharpe_turnover_penalty':
                    turnover = np.sum(np.abs(action - getattr(self, 'last_action', np.zeros_like(action))))
                    reward = mean_ret / std_ret - 0.01 * turnover
                    self.last_action = action.copy()
                else:
                    reward = mean_ret / std_ret  # 默认sharpe
                # 额外加回报激励和风险惩罚
                reward += 0.3 * ret - 0.05 * volatility - 0.05 * max_drawdown
                high_weight_penalty = np.sum(np.maximum(position_ratio - 0.5, 0))
                reward -= 0.1 * high_weight_penalty
                if volatility > np.mean(self.returns_window) + 2 * np.std(self.returns_window):
                    reward -= 0.1 * volatility
                if ret > 0:
                    reward += 0.1 * ret
            else:
                reward = 0.0
        # --- end ---
        self.t += 1
        if not done:
            done = (self.t >= len(self.price_array) - 1)
        state = self._get_state()
        terminated = done
        truncated = False
        info = {'portfolio_value': self.portfolio_value, 'cash': self.cash, 'position': self.position, 'prices': prices}
        return state, reward, terminated, truncated, info

    def _get_state(self):
        # 持仓比例、现金比例 always included
        prices = self.price_array[self.t-1]
        total_value = self.cash + np.sum(self.position * prices)
        position_ratio = (self.position * prices) / (total_value + 1e-8)
        cash_ratio = self.cash / (total_value + 1e-8)
        
        state_components = [position_ratio, [cash_ratio]] # 基础状态分量
        
        if self.include_lstm_features: # 根据参数决定是否包含LSTM特征
            # LSTM特征用多因子特征滑窗
            window = self.feature_array[self.t-self.window_size:self.t]  # (window, N*feature_dim)
            window = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-8)
            X = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, window, N*feature_dim)
            with torch.no_grad():
                feats = self.lstm_model(X).cpu().numpy().flatten()  # (N*feature_dim,)
            state_components.insert(0, feats) # 将LSTM特征插入到前面
            
        state = np.concatenate(state_components) # 拼接状态分量
        
        return state.astype(np.float32) 