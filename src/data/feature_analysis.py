import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 加载数据
X = np.load('data/lstm_X.npy')
y = np.load('data/lstm_y.npy')
print(f'Original feature dimension: {X.shape}')

# 1. 剔除全为0特征列
zero_cols = (X == 0).all(axis=0)
print(f'Number of all-zero feature columns: {zero_cols.sum()} / {X.shape[1]}')
print('全为0特征列索引:', np.where(zero_cols)[0])
X_clean = X[:, ~zero_cols]
print(f'剔除后特征维度: {X_clean.shape}')
np.save('data/lstm_X_clean.npy', X_clean)
print('已保存剔除全为0特征后的lstm_X_clean.npy')

# 2. 可视化部分特征分布
os.makedirs('data/feature_plots', exist_ok=True)
num_plot = min(10, X_clean.shape[1])
for i in range(num_plot):
    plt.figure(figsize=(8,4))
    plt.hist(X_clean[:, i], bins=50, color='skyblue', edgecolor='k')
    plt.title(f'Feature {i} Histogram')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'data/feature_plots/feature_{i}_hist.png')
    plt.close()
    # 箱线图
    plt.figure(figsize=(4,6))
    plt.boxplot(X_clean[:, i], vert=True)
    plt.title(f'Feature {i} Boxplot')
    plt.tight_layout()
    plt.savefig(f'data/feature_plots/feature_{i}_box.png')
    plt.close()
print(f'已保存前{num_plot}个特征的分布图到data/feature_plots/')

# 3. 极值、均值、方差统计
stats = pd.DataFrame({
    'mean': X_clean.mean(axis=0),
    'std': X_clean.std(axis=0),
    'min': X_clean.min(axis=0),
    'max': X_clean.max(axis=0)
})
stats.to_csv('data/feature_stats.csv')
print('已保存特征统计到data/feature_stats.csv')

# 4. 可选：特征与标签相关性分析
corrs = []
for i in range(X_clean.shape[1]):
    try:
        corr = np.corrcoef(X_clean[:, i], y)[0, 1]
    except Exception:
        corr = np.nan
    corrs.append(corr)
pd.Series(corrs).to_csv('data/feature_corrs.csv', header=['corr_with_y'])
print('已保存特征与标签相关性到data/feature_corrs.csv') 