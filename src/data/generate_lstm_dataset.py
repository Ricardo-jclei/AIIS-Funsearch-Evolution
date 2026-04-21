import os
import numpy as np
import pandas as pd
from glob import glob
import random

# 配置
PROCESSED_DIR = 'data/processed'
MARKET_TYPE = 'daily'  # 可改为'minute'
SEQUENCE_LENGTH = 20
# 支持的中英文列名映射
FEATURE_CANDIDATES = [
    ['开盘', 'open'],
    ['收盘', 'close'],
    ['最高', 'high'],
    ['最低', 'low'],
    ['成交量', 'volume'],
    ['成交额', 'amount']
]
LABEL_CANDIDATES = ['收盘', 'close']

# 1. 扫描全量字段
fund_all_cols = set()
senti_all_cols = set()
macro_all_cols = set()
for stock_code in os.listdir(PROCESSED_DIR):
    # 基本面
    fund_path = glob(os.path.join(PROCESSED_DIR, stock_code, 'fundamental', '*', '*.csv'))
    for p in fund_path:
        df = pd.read_csv(p, nrows=1)
        fund_all_cols.update(df.columns)
    # 情绪
    senti_path = glob(os.path.join(PROCESSED_DIR, stock_code, 'sentiment', '*', '*.csv'))
    for p in senti_path:
        df = pd.read_csv(p, nrows=1)
        senti_all_cols.update(df.columns)
# 宏观
macro_path = glob(os.path.join(PROCESSED_DIR, '*', 'macro', '*.csv'))
for p in macro_path:
    df = pd.read_csv(p, nrows=1)
    macro_all_cols.update(df.columns)
fund_all_cols = sorted(list(fund_all_cols - set(['报告日', '报表期', '日期'])))
senti_all_cols = sorted(list(senti_all_cols - set(['日期'])))
macro_all_cols = sorted(list(macro_all_cols - set(['日期'])))

# 2. 读取宏观数据（全局，按日期索引）
macro_df = None
if macro_path:
    macro_df = pd.concat([pd.read_csv(p) for p in macro_path], ignore_index=True)
    if '日期' in macro_df.columns:
        macro_df = macro_df.sort_values('日期').set_index('日期').ffill()

X_all, y_all = [], []
sample_dates = []  # 新增：记录每个样本的末尾日期
date_col_name = '日期'  # 默认日期列名
sample_codes = []  # 新增：记录每个样本的股票代码

for stock_code in os.listdir(PROCESSED_DIR):
    market_dir = os.path.join(PROCESSED_DIR, stock_code, 'market', MARKET_TYPE)
    if not os.path.isdir(market_dir):
        continue
    # 读取基本面（最近一期，按报告日索引）
    fund_path = glob(os.path.join(PROCESSED_DIR, stock_code, 'fundamental', '*', '*.csv'))
    fund_df = None
    if fund_path:
        fund_df = pd.concat([pd.read_csv(p) for p in fund_path], ignore_index=True)
        for col in ['报告日', '报表期', '日期']:
            if col in fund_df.columns:
                fund_df = fund_df.sort_values(col).set_index(col).ffill()
                break
    # 读取情绪（按日期索引）
    senti_path = glob(os.path.join(PROCESSED_DIR, stock_code, 'sentiment', '*', '*.csv'))
    senti_df = None
    if senti_path:
        senti_df = pd.concat([pd.read_csv(p) for p in senti_path], ignore_index=True)
        if '日期' in senti_df.columns:
            senti_df = senti_df.sort_values('日期').set_index('日期').ffill()
    for fname in os.listdir(market_dir):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(market_dir, fname))
        # 自动识别特征列和标签列
        feature_cols = []
        for cands in FEATURE_CANDIDATES:
            for cand in cands:
                if cand in df.columns:
                    feature_cols.append(cand)
                    break
        label_col = None
        for cand in LABEL_CANDIDATES:
            if cand in df.columns:
                label_col = cand
                break
        if len(feature_cols) < 6 or label_col is None:
            continue
        data = df[feature_cols + [label_col]].values
        # 归一化（这里只做简单标准化）
        data = (data - np.nanmean(data, axis=0)) / (np.nanstd(data, axis=0) + 1e-8)
        date_col = '日期' if '日期' in df.columns else 'datetime'
        for i in range(len(data) - SEQUENCE_LENGTH):
            # 市场序列特征
            X_seq = data[i:i+SEQUENCE_LENGTH, :-1]
            # 窗口末尾日期
            cur_date = str(df[date_col].iloc[i+SEQUENCE_LENGTH])
            # 拼接基本面（全量字段对齐）
            X_fund = np.zeros(len(fund_all_cols))
            if fund_df is not None:
                mask = fund_df.index <= cur_date
                if mask.any():
                    fund_row = fund_df.loc[mask].iloc[-1]
                    for idx, col in enumerate(fund_all_cols):
                        if col in fund_row.index:
                            val = fund_row[col]
                            # 仅赋值数值型，否则填0
                            try:
                                X_fund[idx] = float(val) if pd.notnull(val) else 0
                            except Exception:
                                # 输出被跳过的字段名和内容
                                print(f"[Skip non-numeric field] Fundamental: {col}={val}")
                                X_fund[idx] = 0
            # 拼接情绪（全量字段对齐）
            X_senti = np.zeros(len(senti_all_cols))
            if senti_df is not None and cur_date in senti_df.index:
                senti_row = senti_df.loc[cur_date]
                for idx, col in enumerate(senti_all_cols):
                    if col in senti_row.index:
                        val = senti_row[col]
                        try:
                            X_senti[idx] = float(val) if pd.notnull(val) else 0
                        except Exception:
                            print(f"[Skip non-numeric field] Sentiment: {col}={val}")
                            X_senti[idx] = 0
            # 拼接宏观（全量字段对齐）
            X_macro = np.zeros(len(macro_all_cols))
            if macro_df is not None and cur_date in macro_df.index:
                macro_row = macro_df.loc[cur_date]
                for idx, col in enumerate(macro_all_cols):
                    if col in macro_row.index:
                        val = macro_row[col]
                        try:
                            X_macro[idx] = float(val) if pd.notnull(val) else 0
                        except Exception:
                            print(f"[Skip non-numeric field] Macro: {col}={val}")
                            X_macro[idx] = 0
            # 合并所有特征
            X_full = np.concatenate([X_seq.flatten(), X_fund, X_senti, X_macro])
            X_all.append(X_full)
            y_all.append(data[i+SEQUENCE_LENGTH, -1])
            sample_dates.append(cur_date)  # 记录末尾日期
            sample_codes.append(stock_code)  # 记录股票代码

# 统计所有样本的特征长度分布
lengths = [x.shape[0] for x in X_all]
print("特征长度分布：", pd.Series(lengths).value_counts())

X_all = np.array(X_all)
y_all = np.array(y_all)
print(f'融合特征后LSTM训练集：X.shape={X_all.shape}, y.shape={y_all.shape}')
np.save('data/lstm_X.npy', X_all)
np.save('data/lstm_y.npy', y_all)
print('已保存到 data/lstm_X.npy, data/lstm_y.npy')

# ===== 自动校验部分 =====
print('\n===== 自动校验报告 =====')

# 1. 随机抽查5个样本，输出窗口末尾索引、标签值、末尾日期、股票代码，并比对处理后market归一化收盘价
num_samples = min(5, len(X_all))
indices = random.sample(range(len(X_all)), num_samples)
print(f'随机抽查{num_samples}个样本:')
for idx in indices:
    code = sample_codes[idx]
    date = sample_dates[idx]
    label = y_all[idx]
    print(f'  样本索引: {idx}, 股票: {code}, 日期: {date}, 标签y: {label}')
    # 自动比对处理后market收盘价（归一化后）
    market_dir = os.path.join(PROCESSED_DIR, code, 'market', MARKET_TYPE)
    close_val = None
    close_mean, close_std = None, None
    found = False
    for fname in os.listdir(market_dir):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(market_dir, fname))
        if '日期' in df.columns and '收盘' in df.columns:
            row = df[df['日期'].astype(str) == date]
            if not row.empty:
                close_val = row['收盘'].values[0]
                # 计算该market文件的收盘均值和方差
                close_mean = df['收盘'].mean()
                close_std = df['收盘'].std() + 1e-8
                found = True
                break
        elif 'datetime' in df.columns and 'close' in df.columns:
            row = df[df['datetime'].astype(str) == date]
            if not row.empty:
                close_val = row['close'].values[0]
                close_mean = df['close'].mean()
                close_std = df['close'].std() + 1e-8
                found = True
                break
    if found and close_val is not None:
        close_val_norm = (close_val - close_mean) / close_std
        print(f'    Original close price: {close_val}, Normalized: {close_val_norm:.6f}, Label y: {label}, Difference: {abs(close_val_norm - label):.6f}')
        if abs(close_val_norm - label) < 1e-4:
            print('    [OK] 标签与归一化后收盘价一致')
        else:
            print('    [警告] 标签与归一化后收盘价不一致！')
    else:
        print('    [警告] 未找到处理后market收盘价进行比对')

# 2. 检查归一化效果
X = np.load('data/lstm_X.npy')
print(f'特征均值: {X.mean():.4f}, 方差: {X.std():.4f}, 最大值: {X.max():.4f}, 最小值: {X.min():.4f}')

# 3. 检查全为0的特征列数量
zero_cols = (X == 0).all(axis=0)
print(f'Number of all-zero feature columns: {zero_cols.sum()} / {X.shape[1]}')

# 4. 输出训练集/验证集划分建议
split_idx = int(len(X) * 0.8)
print(f'建议划分: 训练集 {split_idx} 条，验证集 {len(X) - split_idx} 条（按时间顺序）')
print('如需自动生成划分文件，可进一步实现。') 