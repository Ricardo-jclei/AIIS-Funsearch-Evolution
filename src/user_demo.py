import numpy as np
import torch
# from src.rl.multi_asset_ddpg_agent import DDPGAgent  # [已屏蔽DDPG]
from src.portfolio.optimizer import optimize_portfolio
from src.eval.metrics import sharpe_ratio, sortino_ratio, max_drawdown
from src.eval.compare import compare_portfolios
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import platform
from src.model.enhanced_lstm import EnhancedLSTMModel as LSTMModel
from stable_baselines3 import PPO
from src.rl.multi_asset_trading_env import MultiAssetTradingEnv
import yaml

# ====== 0. 自动设置matplotlib中文字体，解决乱码 ======
if platform.system() == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # Windows
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']            # Linux
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ====== 1. 加载多因子特征数据 ======
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
# 只保留特征列（去掉日期）
multi_factor_array = mf_df.drop(columns=['日期']).values  # (T, N*因子数)

# ====== 1.1 加载收盘价数据（用于回测） ======
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

# ====== 2. 加载RL模型 ======
# state_dim = 5 + 5 + 1 + 5 * 20
# action_dim = 5
# agent = DDPGAgent(state_dim, action_dim)  # [已屏蔽DDPG]
# agent.actor.load_state_dict(torch.load('model_ckpt/multi_asset_ddpg_actor.pth', map_location='cpu'))  # [已屏蔽DDPG权重加载]

# ====== 2.1 加载LSTM模型（协同推理用） ======
lstm_model_path = 'model_ckpt/best_lstm_multi_asset.pth'
window_size = 20
lstm_input_size = 224
lstm_hidden_size = 128
lstm_num_layers = 3
lstm_output_size = 224  # [自动修正] 与训练时一致，输出多因子特征维度
lstm_model = LSTMModel(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size, 0.3)
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location='cpu'))
lstm_model.eval()

# ====== 2.2 加载PPO模型（协同推理用） ======
# 自动查找reward_compare实验Sharpe最高的权重
ppo_results_path = 'output/ppo_reward_compare/ppo_reward_compare_results.csv'
ppo_model_path = 'model_ckpt/ppo_multi_asset_sharpe_turnover_penalty.zip'  # 默认
if os.path.exists(ppo_results_path):
    df = pd.read_csv(ppo_results_path)
    if 'Sharpe' in df.columns and 'weight_path' in df.columns:
        idx = df['Sharpe'].astype(float).idxmax()
        best_weight = df.loc[idx, 'weight_path']
        if isinstance(best_weight, str) and os.path.exists(best_weight):
            ppo_model_path = best_weight
ppo_model = PPO.load(ppo_model_path)

# ====== 3. 用户输入风险偏好和优化方式 ======
risk_aversion = float(input('请输入风险厌恶系数（如1.0-5.0，数值越大越保守）：'))
print('可选优化方式: rl（RL输出）、lstm_rl（LSTM+RL协同）、equal（等权）、minvar（最小方差）、maxsharpe（最大夏普）')
opt_methods = ['rl', 'lstm_rl', 'equal', 'minvar', 'maxsharpe']

# ====== 4. 多种权重方案 ======
def minvar_weights(price_array):
    # 经典最小方差组合
    returns = (price_array[1:] - price_array[:-1]) / price_array[:-1]
    cov = np.cov(returns.T)
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ np.ones(len(stock_list))
    w = w / np.sum(w)
    return w

def maxsharpe_weights(price_array):
    # 经典最大夏普组合（无风险利率为0）
    returns = (price_array[1:] - price_array[:-1]) / price_array[:-1]
    mean_ret = np.mean(returns, axis=0)
    cov = np.cov(returns.T)
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mean_ret
    w = w / np.sum(w)
    return w

# ====== 4.1 LSTM+RL协同推理权重生成 ======
# def lstm_rl_weights(multi_factor_array, agent, lstm_model, window_size=20):
#     ... # [已屏蔽DDPG相关函数]

# ====== 4.2 LSTM+PPO协同推理权重生成（静态权重，仅用于展示） ======
def lstm_ppo_weights(multi_factor_array, lstm_model, ppo_model, window_size=20):
    N = 5
    position = np.zeros(N)
    cash = 1e7
    weights_list = []
    for t in range(window_size, len(multi_factor_array)):
        window = multi_factor_array[t-window_size:t]
        window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-8)
        X = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            feats = lstm_model(X).cpu().numpy().flatten()  # (224,)
        prices = price_array[t-1]
        total_value = cash + np.sum(position * prices)
        position_ratio = (position * prices) / (total_value + 1e-8)
        cash_ratio = cash / (total_value + 1e-8)
        state = np.concatenate([feats, position_ratio, [cash_ratio]]).astype(np.float32)
        obs = state
        action, _ = ppo_model.predict(obs, deterministic=True)
        w = np.clip(action, 0, 1)
        w = w / (np.sum(w) + 1e-8)
        weights_list.append(w)
    weights = np.mean(np.array(weights_list), axis=0)
    return weights

# ====== 4.3 LSTM+PPO动态RL推理（真实RL策略回测） ======
def lstm_ppo_dynamic_backtest(price_array, feature_array, ppo_model, window_size=20, sharpe_window=20, lstm_input_size=224):
    # 自动读取手续费/滑点配置
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '../src/config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
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
        sharpe_window=sharpe_window,
        reward_type='sharpe_turnover_penalty',
        lstm_input_size=lstm_input_size,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        enable_slippage=enable_slippage
    )
    obs, info = env.reset()
    nav = [env.initial_cash]
    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        nav.append(env.portfolio_value)
        done = terminated or truncated
    nav = np.array(nav)
    returns = (nav[1:] - nav[:-1]) / nav[:-1]
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    calmar = -max_drawdown(nav)
    return nav, sharpe, sortino, calmar

# ====== 4.4 PPO（无LSTM特征）动态RL推理 ======
def ppo_no_lstm_dynamic_backtest(price_array, feature_array, ppo_model, window_size=20, sharpe_window=20):
    # 自动读取手续费/滑点配置
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '../src/config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    fee_rate = config.get('fee_rate', 0.001)
    slippage_rate = config.get('slippage_rate', 0.0)
    enable_slippage = config.get('enable_slippage', False)
    
    # 创建一个新的PPO模型，使用正确的状态空间维度
    state_dim = feature_array.shape[1] + len(stock_list) + 1  # 特征维度 + 持仓比例 + 现金比例
    action_dim = len(stock_list)
    env = MultiAssetTradingEnv(
        price_array=price_array,
        feature_array=feature_array,
        window_size=window_size,
        sharpe_window=sharpe_window,
        reward_type='sharpe_turnover_penalty',
        include_lstm_features=False,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        enable_slippage=enable_slippage
    )
    
    # 重新训练PPO模型
    ppo_model = PPO("MlpPolicy", env, verbose=1)
    ppo_model.learn(total_timesteps=10000)  # 可以根据需要调整训练步数
    
    obs, info = env.reset()
    nav = [env.initial_cash]
    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        nav.append(env.portfolio_value)
        done = terminated or truncated
    nav = np.array(nav)
    returns = (nav[1:] - nav[:-1]) / nav[:-1]
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    calmar = -max_drawdown(nav)
    return nav, sharpe, sortino, calmar

weights_dict = {}
# weights_dict['RL'] = optimize_portfolio(agent, multi_factor_array, risk_aversion, method='rl', window_size=20)  # [已屏蔽DDPG]
# weights_dict['LSTM+RL'] = lstm_rl_weights(multi_factor_array, agent, lstm_model, window_size=20)  # [已屏蔽DDPG]
weights_dict['等权'] = np.ones(len(stock_list)) / len(stock_list)
weights_dict['最小方差'] = minvar_weights(price_array)
weights_dict['最大夏普'] = maxsharpe_weights(price_array)
weights_dict['LSTM+PPO'] = lstm_ppo_weights(multi_factor_array, lstm_model, ppo_model, window_size=20)

# ====== 5. 权重美化输出 ======
weights_df = pd.DataFrame(weights_dict, index=stock_list)
print('\nWeight distribution of each strategy (LSTM+PPO based on reward: sharpe_turnover_penalty, static weights):')
print(weights_df.round(4))

# ====== 5.5 静态权重回测函数补充 ======
def backtest(weights, price_array, initial_cash=1e7):
    nav = [initial_cash]
    for t in range(1, len(price_array)):
        ret = np.dot((price_array[t] - price_array[t-1]) / price_array[t-1], weights)
        nav.append(nav[-1] * (1 + ret))
    return np.array(nav)

# ====== 6. 回测净值 ======
nav_dict = {}
for name, w in weights_dict.items():
    nav_dict[name] = backtest(w, price_array)
# LSTM+PPO动态RL推理净值
lstmppo_nav, lstmppo_sharpe, lstmppo_sortino, lstmppo_calmar = lstm_ppo_dynamic_backtest(price_array, multi_factor_array, ppo_model, window_size=20, sharpe_window=20, lstm_input_size=multi_factor_array.shape[1])
nav_dict['LSTM+PPO动态RL'] = lstmppo_nav
# PPO（无LSTM特征）动态RL推理净值
ppo_no_lstm_nav, ppo_no_lstm_sharpe, ppo_no_lstm_sortino, ppo_no_lstm_calmar = ppo_no_lstm_dynamic_backtest(price_array, multi_factor_array, ppo_model, window_size=20, sharpe_window=20)
nav_dict['PPO动态RL(无LSTM)'] = ppo_no_lstm_nav

# ====== 6.1 LSTM+PPO动态RL权重热力图可视化 ======
output_dir = 'output/compare'
os.makedirs(output_dir, exist_ok=True)
# 记录每步权重
lstmppo_dynamic_weights = []
env = MultiAssetTradingEnv(
    price_array=price_array,
    feature_array=multi_factor_array,
    window_size=20,
    sharpe_window=20,
    reward_type='sharpe_turnover_penalty',
    lstm_input_size=multi_factor_array.shape[1]
)
obs, info = env.reset()
done = False
while not done:
    action, _ = ppo_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    lstmppo_dynamic_weights.append(action)
    done = terminated or truncated
lstmppo_dynamic_weights = np.array(lstmppo_dynamic_weights)  # (T, N)
# 绘制热力图
plt.figure(figsize=(12, 4))
plt.imshow(lstmppo_dynamic_weights.T, aspect='auto', cmap='YlGnBu')
plt.colorbar(label='权重')
plt.yticks(range(len(stock_list)), stock_list)
plt.xlabel('时间步', fontsize=13)
plt.ylabel('资产', fontsize=13)
plt.title('LSTM+PPO动态RL权重热力图', fontsize=15)
plt.tight_layout()
plt.savefig(f'{output_dir}/lstmppo_dynamic_weights_heatmap.png', dpi=180)
plt.close()

# ====== 7. 绩效评估与对比（表格） ======
result = compare_portfolios(nav_dict)
result_df = pd.DataFrame(result).T
# 用动态RL推理的真实指标覆盖静态LSTM+PPO
result_df.loc['LSTM+PPO动态RL', ['Sharpe', 'Sortino', 'MaxDrawdown']] = [lstmppo_sharpe, lstmppo_sortino, lstmppo_calmar]
# result_df.loc['PPO动态RL(无LSTM)', ['Sharpe', 'Sortino', 'MaxDrawdown']] = [ppo_no_lstm_sharpe, ppo_no_lstm_sortino, ppo_no_lstm_calmar]
print('\n绩效指标对比 (LSTM+PPO基于reward: sharpe_turnover_penalty, 动态RL为真实表现):')
print(result_df.round(4))

# ====== 8. 可视化（单独保存到 output/compare/） ======
# 策略颜色映射
color_map = {
    '等权': '#1f77b4',           # 蓝色
    '最小方差': '#2ca02c',       # 绿色
    '最大夏普': '#d62728',       # 红色
    'LSTM+PPO': '#7f7f7f',       # 灰色
    'LSTM+PPO动态RL': '#ff7f0e', # 橙色
    'PPO动态RL(无LSTM)': '#800080', # 紫色，用于区分无LSTM的情况
    '市场指数': '#000000'         # 黑色
}
# 净值曲线
plt.figure(figsize=(12,7))
for name, nav in nav_dict.items():
    plt.plot(nav, label=name, linewidth=2, color=color_map.get(name, None))
# 自动加载市场指数净值（如有）
market_index_path = 'data/market_index_nav.csv'
if os.path.exists(market_index_path):
    market_index_nav = pd.read_csv(market_index_path).values.squeeze()
    plt.plot(market_index_nav[:min(len(market_index_nav), len(nav))], label='市场指数', color=color_map['市场指数'], linestyle='--', linewidth=2)
plt.legend(fontsize=13, title="策略/资产类别")
plt.xlabel('时间步', fontsize=14)
plt.ylabel('净值', fontsize=14)
plt.title('各方案净值曲线', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/portfolio_nav_compare.png', dpi=180)
plt.close()
# 权重分布
ax = weights_df.plot(kind='bar', figsize=(12,7), width=0.8, color=[color_map.get(c, None) for c in weights_df.columns])
plt.title('各方案权重分布', fontsize=16)
plt.ylabel('权重', fontsize=14)
plt.xlabel('资产', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=13, title="策略/资产类别")
plt.tight_layout()
plt.savefig(f'{output_dir}/portfolio_weights_compare.png', dpi=180)
plt.close()
# 回撤曲线
plt.figure(figsize=(12,7))
for name, nav in nav_dict.items():
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / (peak + 1e-8)
    plt.plot(drawdown, label=name, linewidth=2, color=color_map.get(name, None))
plt.legend(fontsize=13, title="策略/资产类别")
plt.xlabel('时间步', fontsize=14)
plt.ylabel('回撤率', fontsize=14)
plt.title('各方案回撤曲线', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/portfolio_drawdown_compare.png', dpi=180)
plt.close()

# ====== 9. 自动生成markdown报告和HTML报告（含图片说明和颜色对照） ======
report_md_path = f'{output_dir}/portfolio_report.md'
with open(report_md_path, 'w', encoding='utf-8') as f:
    f.write('# 投资组合优化与绩效评估报告\n')
    f.write('## 策略颜色对照\n')
    f.write('| 策略 | 颜色 |\n|---|---|\n')
    for k, v in color_map.items():
        f.write(f'| {k} | <span style="color:{v}">{v}</span> |\n')
    f.write('## 权重分布\n')
    f.write(weights_df.round(4).to_markdown() + '\n')
    f.write(f'![权重分布](portfolio_weights_compare.png)\n')
    f.write('## LSTM+PPO动态RL权重热力图\n')
    f.write(f'![LSTM+PPO动态RL权重热力图](lstmppo_dynamic_weights_heatmap.png)\n')
    f.write('## 净值曲线\n')
    f.write(f'![净值曲线](portfolio_nav_compare.png)\n')
    f.write('## 回撤曲线\n')
    f.write(f'![回撤曲线](portfolio_drawdown_compare.png)\n')
    f.write('## 绩效指标对比\n')
    f.write(result_df.round(4).to_markdown() + '\n')
print(f'\n已自动生成投资组合优化报告: {report_md_path}')

# HTML报告
report_html_path = f'{output_dir}/portfolio_report.html'
with open(report_html_path, 'w', encoding='utf-8') as f:
    f.write('<html><head><meta charset="utf-8"><title>投资组合优化与绩效评估报告</title></head><body>')
    f.write('<h1>投资组合优化与绩效评估报告</h1>')
    f.write('<h2>策略颜色对照</h2>')
    f.write('<table border="1"><tr><th>策略</th><th>颜色</th></tr>')
    for k, v in color_map.items():
        f.write(f'<tr><td>{k}</td><td><span style="color:{v}">{v}</span></td></tr>')
    f.write('</table>')
    f.write('<h2>权重分布</h2>')
    f.write(weights_df.round(4).to_html())
    f.write(f'<img src="portfolio_weights_compare.png" width="700"/><br>')
    f.write('<h2>LSTM+PPO动态RL权重热力图</h2>')
    f.write(f'<img src="lstmppo_dynamic_weights_heatmap.png" width="700"/><br>')
    f.write('<h2>净值曲线</h2>')
    f.write(f'<img src="portfolio_nav_compare.png" width="700"/><br>')
    f.write('<h2>回撤曲线</h2>')
    f.write(f'<img src="portfolio_drawdown_compare.png" width="700"/><br>')
    f.write('<h2>绩效指标对比</h2>')
    f.write(result_df.round(4).to_html())
    # 预留交互式可视化接口
    f.write('<h2>交互式可视化（预留）</h2>')
    f.write('<p>后续可集成Plotly、Bokeh等交互式图表。</p>')
    f.write('</body></html>')
print(f'已自动生成HTML报告: {report_html_path}') 