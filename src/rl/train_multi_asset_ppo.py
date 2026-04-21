import numpy as np
import pandas as pd
import torch
import os
from src.rl.multi_asset_trading_env import MultiAssetTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import yaml
import matplotlib
import platform

if platform.system() == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # Windows
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']            # Linux
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ====== 1. 股票池与数据路径 ======
stock_list = [
    '600519', '600030', '600036', '601318', '601988'
]
data_dir_tpl = 'data/processed/{}/multi_factor.csv'

# ====== 2. 读取并对齐多资产多因子特征 ======
feature_dfs = []
price_dfs = []
for code in stock_list:
    path = data_dir_tpl.format(code)
    df = pd.read_csv(path)
    df = df.rename(columns={col: f'{col}_{code}' for col in df.columns if col != '日期'})
    feature_dfs.append(df)
    price_col = [col for col in df.columns if ('收盘' in col or 'close' in col.lower()) and code in col]
    if price_col:
        price_dfs.append(df[['日期', price_col[0]]].rename(columns={price_col[0]: code}))
    else:
        raise ValueError(f'未找到{code}的收盘价列！')
feature_df = feature_dfs[0]
for df in feature_dfs[1:]:
    feature_df = feature_df.merge(df, on='日期', how='inner')
feature_cols = [col for col in feature_df.columns if col != '日期']
feature_array = feature_df[feature_cols].values
price_df = price_dfs[0]
for df in price_dfs[1:]:
    price_df = price_df.merge(df, on='日期', how='inner')
price_array = price_df[stock_list].values
print(f"[Data] Multi-asset multi-factor feature shape: {feature_array.shape}, Price shape: {price_array.shape}")

# ====== 3. 环境参数 ======
window_size = 20
initial_cash = 1e7
lstm_input_size = feature_array.shape[1]
lstm_model_path = 'model_ckpt/best_lstm_multi_asset.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
asset_num = len(stock_list)
sharpe_window = 20
reward_type = 'sharpe_turnover_penalty'

# ====== 3.1 读取手续费/滑点配置 ======
config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
else:
    config = {}
fee_rate = config.get('fee_rate', 0.001)
slippage_rate = config.get('slippage_rate', 0.0)
enable_slippage = config.get('enable_slippage', False)

env = MultiAssetTradingEnv(
    price_array=price_array,
    feature_array=feature_array,
    window_size=window_size,
    initial_cash=initial_cash,
    device=device,
    lstm_model_path=lstm_model_path,
    lstm_input_size=lstm_input_size,
    asset_num=asset_num,
    fee_rate=fee_rate,
    slippage_rate=slippage_rate,
    enable_slippage=enable_slippage,
    sharpe_window=sharpe_window,
    reward_type=reward_type,
    include_lstm_features=True
)

# ====== 4. 检查环境 ======
check_env(env, warn=True)

# ====== 5. PPO训练 ======
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/", n_steps=2048, batch_size=64, learning_rate=1e-3, n_epochs=10)
model.learn(total_timesteps=200_000)
model.save('model_ckpt/ppo_multi_asset_sharpe_turnover_penalty.zip')
print(f'PPO model saved, reward_type={reward_type}')

# ====== 6. 测试与可视化 ======
color_map = {
    '600519': '#1f77b4',
    '600030': '#2ca02c',
    '600036': '#d62728',
    '601318': '#9467bd',
    '601988': '#8c564b',
    '市场指数': '#000000'
}
obs, info = env.reset()
portfolio_values = []
rewards = []
position_ratios = []
for _ in range(len(price_array) - window_size - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    portfolio_values.append(info['portfolio_value'])
    rewards.append(reward)
    # 记录持仓权重
    prices = info['prices']
    position = info['position']
    total_value = info['cash'] + np.sum(position * prices)
    position_ratio = (position * prices) / (total_value + 1e-8)
    position_ratios.append(position_ratio)
    done = terminated or truncated
    if done:
        break
position_ratios = np.array(position_ratios)  # shape: (T, asset_num)
# 可视化持仓权重变化
plt.figure(figsize=(10, 6))
for i, code in enumerate(stock_list):
    plt.plot(position_ratios[:, i], label=code, color=color_map.get(code, None))
plt.legend(title="Stock Code")
plt.xlabel('Step')
plt.ylabel('Position Ratio')
plt.title('Asset Weight Dynamics')
plt.savefig('model_ckpt/ppo_multi_asset_position_ratio_curve.png')
plt.close()
plt.figure()
plt.plot(portfolio_values, label='PPO策略', color='#0072B2', linewidth=2)
# 自动加载市场指数净值（如有）
market_index_path = 'data/market_index_nav.csv'
if os.path.exists(market_index_path):
    market_index_nav = pd.read_csv(market_index_path).values.squeeze()
    plt.plot(market_index_nav[:len(portfolio_values)], label='Market Index', color=color_map['市场指数'], linestyle='--', linewidth=2)
plt.legend(title="Strategy/Asset Class")
plt.xlabel('Step')
plt.ylabel('Portfolio Value')
plt.title('PPO Portfolio Value Curve (Sharpe Reward)')
plt.legend(title="策略/资产类别")
plt.savefig('model_ckpt/ppo_multi_asset_sharpe_portfolio_curve.png')
plt.close()
plt.figure()
plt.plot(rewards, label='Sharpe Reward', color='#d62728')
plt.xlabel('Step')
plt.ylabel('Sharpe Reward')
plt.title('PPO Sharpe Reward Curve')
plt.legend()
plt.savefig('model_ckpt/ppo_multi_asset_sharpe_reward_curve.png')
plt.close()
print('PPO multi-asset Sharpe ratio reward training and evaluation completed, results saved to model_ckpt/') 