import numpy as np

# 加载数据
X = np.load('data/lstm_X_clean.npy')
y = np.load('data/lstm_y.npy')

# 按时间顺序划分
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)
train_X, val_X = X[:split_idx], X[split_idx:]
train_y, val_y = y[:split_idx], y[split_idx:]

np.save('data/train_X.npy', train_X)
np.save('data/val_X.npy', val_X)
np.save('data/train_y.npy', train_y)
np.save('data/val_y.npy', val_y)

print(f'训练集: train_X.npy {train_X.shape}, train_y.npy {train_y.shape}')
print(f'验证集: val_X.npy {val_X.shape}, val_y.npy {val_y.shape}')
print('已完成训练/验证集划分并保存。') 