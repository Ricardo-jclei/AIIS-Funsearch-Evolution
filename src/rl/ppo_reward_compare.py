import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from src.rl.multi_asset_trading_env import MultiAssetTradingEnv
from src.eval.metrics import sharpe_ratio, sortino_ratio, max_drawdown
import matplotlib.pyplot as plt
import matplotlib
import platform

if platform.system() == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # Windows
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']            # Linux
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# === 新增：自动加载多资产多因子特征和价格数据 ===
stock_list = ['600519', '600030', '600036', '601318', '601988']
mf_data_dir_tpl = 'data/processed/{}/multi_factor.csv'
mf_dfs = []
for code in stock_list:
    path = mf_data_dir_tpl.format(code)
    df = pd.read_csv(path)
    df = df.rename(columns={col: f'{code}_{col}' for col in df.columns if col != '日期'})
    mf_dfs.append(df)
mf_df = mf_dfs[0]
for df in mf_dfs[1:]:
    mf_df = mf_df.merge(df, on='日期', how='inner')
feature_array = mf_df.drop(columns=['日期']).values  # (T, N*因子数)
lstm_input_size = feature_array.shape[1]

data_dir_tpl = 'data/processed/{}/market/daily/20220425_20250424_processed.csv'
price_dfs = []
for code in stock_list:
    path = data_dir_tpl.format(code)
    df = pd.read_csv(path, usecols=['日期', '收盘'])
    df = df.rename(columns={'收盘': code})
    price_dfs.append(df)
price_df = price_dfs[0]
for df in price_dfs[1:]:
    price_df = price_df.merge(df, on='日期', how='inner')
price_array = price_df[stock_list].values  # (T, N)

# 1. 环境和数据参数
window_size = 20
sharpe_window = 20
output_dir = 'output/ppo_reward_compare'
os.makedirs(output_dir, exist_ok=True)

reward_types = ['sharpe', 'sortino', 'calmar', 'sharpe_turnover_penalty']
results = {}

# 2. 评估函数
def evaluate_model(model, env):
    obs, _ = env.reset()
    nav = [env.initial_cash]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        nav.append(env.portfolio_value)
    nav = np.array(nav)
    returns = (nav[1:] - nav[:-1]) / nav[:-1]
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    calmar = -max_drawdown(nav)
    return sharpe, sortino, calmar, nav

# 3. 多reward训练与评估
for reward_type in reward_types:
    print(f'训练reward: {reward_type}')
    env = MultiAssetTradingEnv(price_array=price_array, feature_array=feature_array, window_size=window_size, sharpe_window=sharpe_window, reward_type=reward_type, lstm_input_size=lstm_input_size)
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=100_000)
    sharpe, sortino, calmar, nav = evaluate_model(model, env)
    # === 新增：保存当前reward_type下的PPO权重 ===
    ckpt_dir = 'model_ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)
    weight_path = f'{ckpt_dir}/ppo_multi_asset_{reward_type}_best.zip'
    model.save(weight_path)
    results[reward_type] = {'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar, 'nav': nav, 'weight_path': weight_path}

# 4. 输出对比表格
result_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != 'nav'} for k, v in results.items()})
result_df = result_df.T
result_df.to_csv(f'{output_dir}/ppo_reward_compare_results.csv')
print('各reward绩效对比:')
print(result_df.round(4))

# 5. 可视化
plt.figure(figsize=(10,6))
for k, v in results.items():
    plt.plot(v['nav'], label=k)
plt.legend()
plt.xlabel('时间步')
plt.ylabel('净值')
plt.title('不同reward下PPO净值曲线')
plt.tight_layout()
plt.savefig(f'{output_dir}/ppo_reward_compare_nav.png', dpi=150)
plt.close()

plt.figure(figsize=(8,5))
result_df[['Sharpe', 'Sortino', 'Calmar']].plot(kind='bar')
plt.title('不同reward下PPO绩效对比')
plt.ylabel('指标值')
plt.tight_layout()
plt.savefig(f'{output_dir}/ppo_reward_compare_metrics.png', dpi=150)
plt.close()
print(f'对比结果已保存到 {output_dir}/ppo_reward_compare_results.csv')
print(f'净值曲线图已保存到 {output_dir}/ppo_reward_compare_nav.png')
print(f'绩效对比图已保存到 {output_dir}/ppo_reward_compare_metrics.png') 