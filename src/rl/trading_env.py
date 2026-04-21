import gym
import numpy as np
import torch
from gym import spaces
import pandas as pd
import os
from glob import glob
from src.model.lstm_train import LSTMModel

class TradingEnv(gym.Env):
    def __init__(self, price_array, window_size=20, initial_cash=1e6, device='cpu', lstm_model_path='model_ckpt/best_lstm.pth', lstm_input_size=511, processed_dir='data/processed', stock_code='600519', market_type='daily'):
        super().__init__()
        self.price_array = price_array  # shape: (T, F)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.device = device
        self.lstm_input_size = lstm_input_size
        self.processed_dir = processed_dir
        self.stock_code = stock_code
        self.market_type = market_type
        # ==== 1. 扫描全量字段（与generate_lstm_dataset.py一致） ====
        self.fund_all_cols = set()
        self.senti_all_cols = set()
        self.macro_all_cols = set()
        # 基本面
        fund_path = glob(os.path.join(processed_dir, stock_code, 'fundamental', '*', '*.csv'))
        for p in fund_path:
            df = pd.read_csv(p, nrows=1)
            self.fund_all_cols.update(df.columns)
        # 情绪
        senti_path = glob(os.path.join(processed_dir, stock_code, 'sentiment', '*', '*.csv'))
        for p in senti_path:
            df = pd.read_csv(p, nrows=1)
            self.senti_all_cols.update(df.columns)
        # 宏观
        macro_path = glob(os.path.join(processed_dir, '*', 'macro', '*.csv'))
        for p in macro_path:
            df = pd.read_csv(p, nrows=1)
            self.macro_all_cols.update(df.columns)
        self.fund_all_cols = sorted(list(self.fund_all_cols - set(['报告日', '报表期', '日期'])))
        self.senti_all_cols = sorted(list(self.senti_all_cols - set(['日期'])))
        self.macro_all_cols = sorted(list(self.macro_all_cols - set(['日期'])))
        # ==== 2. 预加载多源数据 ====
        # 基本面
        self.fund_df = None
        if fund_path:
            self.fund_df = pd.concat([pd.read_csv(p) for p in fund_path], ignore_index=True)
            for col in ['报告日', '报表期', '日期']:
                if col in self.fund_df.columns:
                    self.fund_df = self.fund_df.sort_values(col).set_index(col).ffill()
                    break
        # 情绪
        self.senti_df = None
        if senti_path:
            self.senti_df = pd.concat([pd.read_csv(p) for p in senti_path], ignore_index=True)
            if '日期' in self.senti_df.columns:
                self.senti_df = self.senti_df.sort_values('日期').set_index('日期').ffill()
        # 宏观
        self.macro_df = None
        if macro_path:
            self.macro_df = pd.concat([pd.read_csv(p) for p in macro_path], ignore_index=True)
            if '日期' in self.macro_df.columns:
                self.macro_df = self.macro_df.sort_values('日期').set_index('日期').ffill()
        # ==== 3. 市场滑窗归一化参数 ====
        self.market_mean = np.nanmean(price_array, axis=0)
        self.market_std = np.nanstd(price_array, axis=0) + 1e-8
        # ==== 4. 状态空间: LSTM特征+持仓+现金 ====
        self.lstm_output_size = 1  # 与LSTMModel output_size一致
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.lstm_output_size+2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # ==== 5. 加载LSTM模型 ====
        self.lstm_model = LSTMModel(lstm_input_size, 64, 2, 1, 0.2, 'tanh').to(device)
        self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        self.lstm_model.eval()
        self.reset()

    def reset(self):
        self.t = self.window_size
        self.cash = self.initial_cash
        self.position = 0.0  # 持仓股数
        self.history = []
        self.debug_info = []  # 新增：收集调试信息
        # ====== 自动初始买入30%仓位，留70%现金 ======
        price = self.price_array[self.t-1, 0]
        if price > 0:
            init_position = (self.initial_cash / price) * 0.3
            self.position = init_position
            self.cash = self.initial_cash - init_position * price
        print(f"[reset] 初始position={self.position}, cash={self.cash}, price={price}")
        return self._get_state()

    def step(self, action, add_noise=True):
        print(f"[step前] t={self.t}, position={self.position}, cash={self.cash}")
        assert self.position >= 0 and self.cash >= 0, "position或cash小于0，reset-step初始化不一致！"
        action = np.clip(action, -1, 1)[0]
        # ====== 训练时action加大噪声增强探索 ======
        if add_noise:
            noise = np.random.normal(0, 0.2)
            action = np.clip(action + noise, -1, 1)
        price = self.price_array[self.t, 0]  # 假设第0列为收盘价
        max_buy = self.cash / price if price > 0 else 0.0
        max_sell = self.position
        buy_amount = 0.0
        sell_amount = 0.0
        fee_rate = 0.002
        fee = 0.0
        if action > 0:
            buy_amount = min(action * max_buy, max_buy)
            fee = buy_amount * price * fee_rate
            if self.cash - buy_amount * price - fee < 0:
                buy_amount = max((self.cash - 1e-6) / (price * (1 + fee_rate)), 0.0)
                fee = buy_amount * price * fee_rate
            self.position += buy_amount
            self.cash += -buy_amount * price - fee
        elif action < 0:
            sell_amount = min(-action * max_sell, max_sell)
            fee = sell_amount * price * fee_rate
            self.position -= sell_amount
            self.cash += sell_amount * price - fee
        # ====== cash/position下限保护 ======
        self.cash = np.clip(self.cash, 0.0, 1e12)
        self.position = np.clip(self.position, 0.0, 1e8)
        if abs(self.cash) < 1e-6:
            self.cash = 0.0
        if abs(self.position) < 1e-6:
            self.position = 0.0
        print(f"[step后] t={self.t}, position={self.position}, cash={self.cash}")
        # 计算奖励
        next_value = self.cash + self.position * price
        prev_value = self.cash + self.position * self.price_array[self.t-1, 0]
        # === 交易成本显式纳入reward ===
        asset_delta = next_value - prev_value
        reward_raw = (asset_delta - fee) / (prev_value + 1e-8)
        # === reward归一化/clip ===
        reward = np.log1p(reward_raw) if reward_raw > -1 else -1.0
        reward = np.clip(reward, -1, 1)
        # === portfolio value归一化/clip ===
        portfolio_value = np.clip(next_value, 0, 1e12)
        portfolio_value_log = np.log1p(portfolio_value)
        self.t += 1
        done = (self.t >= len(self.price_array)-1)
        state = self._get_state()
        info = {'cash': self.cash, 'position': self.position, 'portfolio_value': portfolio_value, 'portfolio_value_log': portfolio_value_log}
        # 收集调试信息
        self.debug_info.append({'step': self.t, 'price': price, 'action': action, 'buy_amount': buy_amount, 'sell_amount': sell_amount, 'position': self.position, 'cash': self.cash, 'reward': reward, 'portfolio_value': portfolio_value})
        # === 每步输出日志 ===
        print(f"[日志] t={self.t}, action={action:.4f}, buy={buy_amount:.2f}, sell={sell_amount:.2f}, reward={reward:.4f}, portfolio_value={portfolio_value:.2f}")
        return state, reward, done, info

    def _get_state(self):
        # 取历史窗口数据，做归一化
        window = self.price_array[self.t-self.window_size:self.t]
        window = (window - self.market_mean) / self.market_std
        # 仅首次打印归一化window
        if not hasattr(self, '_printed_window_debug'):
            print("归一化后window示例：", window)
            self._printed_window_debug = True
        # 取当前日期
        cur_date = str(self.t)
        # ==== 基本面特征 ====
        X_fund = np.zeros(len(self.fund_all_cols))
        if self.fund_df is not None and cur_date in self.fund_df.index:
            fund_row = self.fund_df.loc[cur_date]
            for idx, col in enumerate(self.fund_all_cols):
                if col in fund_row.index:
                    val = fund_row[col]
                    try:
                        X_fund[idx] = float(val) if pd.notnull(val) else 0
                    except Exception:
                        X_fund[idx] = 0
        # ==== 情绪特征 ====
        X_senti = np.zeros(len(self.senti_all_cols))
        if self.senti_df is not None and cur_date in self.senti_df.index:
            senti_row = self.senti_df.loc[cur_date]
            for idx, col in enumerate(self.senti_all_cols):
                if col in senti_row.index:
                    val = senti_row[col]
                    try:
                        X_senti[idx] = float(val) if pd.notnull(val) else 0
                    except Exception:
                        X_senti[idx] = 0
        # ==== 宏观特征 ====
        X_macro = np.zeros(len(self.macro_all_cols))
        if self.macro_df is not None and cur_date in self.macro_df.index:
            macro_row = self.macro_df.loc[cur_date]
            for idx, col in enumerate(self.macro_all_cols):
                if col in macro_row.index:
                    val = macro_row[col]
                    try:
                        X_macro[idx] = float(val) if pd.notnull(val) else 0
                    except Exception:
                        X_macro[idx] = 0
        # ==== 融合所有特征 ====
        X_seq = window.flatten()
        # 检查并修正window、X_fund、X_senti、X_macro
        if np.isnan(X_seq).any() or np.isinf(X_seq).any():
            print("[LSTM输入][警告] window含nan/inf，自动用0填充！")
            X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(X_fund).any() or np.isinf(X_fund).any():
            print("[LSTM输入][警告] X_fund含nan/inf，自动用0填充！")
            X_fund = np.nan_to_num(X_fund, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(X_senti).any() or np.isinf(X_senti).any():
            print("[LSTM输入][警告] X_senti含nan/inf，自动用0填充！")
            X_senti = np.nan_to_num(X_senti, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(X_macro).any() or np.isinf(X_macro).any():
            print("[LSTM输入][警告] X_macro含nan/inf，自动用0填充！")
            X_macro = np.nan_to_num(X_macro, nan=0.0, posinf=0.0, neginf=0.0)
        X_full = np.concatenate([X_seq, X_fund, X_senti, X_macro])
        # 补齐到lstm_input_size
        if X_full.shape[0] < self.lstm_input_size:
            X_full = np.concatenate([X_full, np.zeros(self.lstm_input_size - X_full.shape[0])])
        elif X_full.shape[0] > self.lstm_input_size:
            X_full = X_full[:self.lstm_input_size]
        # === LSTM输入调试 ===
        print(f"[LSTM Input] X_full: max={np.nanmax(X_full):.4e}, min={np.nanmin(X_full):.4e}, mean={np.nanmean(X_full):.4e}, nan={np.isnan(X_full).any()}, inf={np.isinf(X_full).any()}")
        print(f"[LSTM Input] window contains nan/inf: nan={np.isnan(X_seq).any()}, inf={np.isinf(X_seq).any()}")
        print(f"[LSTM Input] X_fund contains nan/inf: nan={np.isnan(X_fund).any()}, inf={np.isinf(X_fund).any()}")
        print(f"[LSTM Input] X_senti contains nan/inf: nan={np.isnan(X_senti).any()}, inf={np.isinf(X_senti).any()}")
        print(f"[LSTM Input] X_macro contains nan/inf: nan={np.isnan(X_macro).any()}, inf={np.isinf(X_macro).any()}")
        # 送入LSTM
        X_full_tensor = torch.tensor(X_full, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lstm_feat = self.lstm_model(X_full_tensor).cpu().numpy().flatten()
        # === 自动归一化position和cash ===
        price = self.price_array[self.t-1, 0]
        max_position = self.initial_cash / price if price > 0 else 1.0
        position_norm = self.position / max_position
        cash_norm = self.cash / self.initial_cash
        state = np.concatenate([lstm_feat, [position_norm], [cash_norm]])
        if not hasattr(self, '_printed_state_debug'):
            print(f"Normalized state example: {state}")
            self._printed_state_debug = True
        return state.astype(np.float32) 