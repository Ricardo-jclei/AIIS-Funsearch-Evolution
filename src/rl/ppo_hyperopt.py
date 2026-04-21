import optuna
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from src.rl.multi_asset_trading_env import MultiAssetTradingEnv
from src.eval.metrics import sharpe_ratio
import matplotlib.pyplot as plt
import matplotlib
import platform

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
reward_type = 'sharpe'  # 固定为sharpe
output_dir = 'output/ppo_hyperopt'
os.makedirs(output_dir, exist_ok=True)

# 2. 评估函数
def evaluate_model(model, env, eval_episodes=1):
    obs, _ = env.reset()
    navs = []
    for _ in range(eval_episodes):
        done = False
        nav = [env.initial_cash]
        obs, _ = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            nav.append(env.portfolio_value)
        navs.append(nav)
    navs = np.array(navs)
    returns = (np.array(navs)[:,1:] - np.array(navs)[:,:-1]) / np.array(navs)[:,:-1]
    sharpe = np.mean([sharpe_ratio(r) for r in returns])
    return sharpe

# 3. Optuna目标函数
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512])
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    env = MultiAssetTradingEnv(price_array=price_array, feature_array=feature_array, window_size=window_size, sharpe_window=sharpe_window, reward_type=reward_type, lstm_input_size=lstm_input_size)
    model = PPO('MlpPolicy', env, learning_rate=lr, n_steps=n_steps, clip_range=clip_range, verbose=0)
    model.learn(total_timesteps=100_000)
    sharpe = evaluate_model(model, env)
    trial.set_user_attr('sharpe', sharpe)
    return sharpe

if __name__ == '__main__':
    # 设置matplotlib中文字体
    if platform.system() == 'Darwin':
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
    elif platform.system() == 'Windows':
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # Windows
    else:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']            # Linux
    matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    print('最优参数:', study.best_params)
    # 保存所有实验结果
    df = study.trials_dataframe()
    df.to_csv(f'{output_dir}/ppo_hyperopt_results.csv', index=False)
    # 可视化
    plt.figure(figsize=(8,5))
    plt.plot(df['value'], marker='o')
    plt.xlabel('Trial')
    plt.ylabel('Sharpe')
    plt.title('PPO超参数搜索Sharpe对比')
    plt.savefig(f'{output_dir}/ppo_hyperopt_sharpe.png', dpi=150)
    plt.close()
    print(f'所有实验结果已保存到 {output_dir}/ppo_hyperopt_results.csv')
    print(f'对比图已保存到 {output_dir}/ppo_hyperopt_sharpe.png')
