import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from src.model.enhanced_lstm import EnhancedLSTMModel
import matplotlib
import platform

if platform.system() == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # Windows
else:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']            # Linux
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ====== 1. 读取多资产多因子特征数据 ======
stock_list = [
    '600519', '600030', '600036', '601318', '601988'
]
data_dir_tpl = 'data/processed/{}/multi_factor.csv'
feature_dfs = []
for code in stock_list:
    path = data_dir_tpl.format(code)
    df = pd.read_csv(path)
    df = df.rename(columns={col: f'{col}_{code}' for col in df.columns if col != '日期'})
    feature_dfs.append(df)
# 按日期对齐
feature_df = feature_dfs[0]
for df in feature_dfs[1:]:
    feature_df = feature_df.merge(df, on='日期', how='inner')
# 取所有资产的特征列
feature_cols = [col for col in feature_df.columns if col != '日期']
feature_array = feature_df[feature_cols].values  # (T, N*feature_dim)
print(f"[LSTM训练] 多资产多因子特征 shape: {feature_array.shape}")

# ====== 2. 构造LSTM训练集 ======
window_size = 20
X, y = [], []
for t in range(window_size, len(feature_array)):
    X.append(feature_array[t-window_size:t])  # shape: (window, N*feature_dim)
    y.append(feature_array[t])                # shape: (N*feature_dim,)
X = np.array(X)  # (样本数, window, N*feature_dim)
y = np.array(y)  # (样本数, N*feature_dim)
print(f"[LSTM训练] X shape: {X.shape}, y shape: {y.shape}")

# 归一化（全局）
X_mean = X.mean(axis=(0,1), keepdims=True)
X_std = X.std(axis=(0,1), keepdims=True) + 1e-8
X_norm = (X - X_mean) / X_std
y_norm = (y - X_mean.squeeze()) / X_std.squeeze()

# 划分训练集/验证集
X_train, X_val, y_train, y_val = train_test_split(X_norm, y_norm, test_size=0.1, random_state=42)

# ====== 3. 定义增强版LSTM模型 ======
# class EnhancedLSTMModel(nn.Module):
#     ... (原模型定义全部删除)

# 初始化模型
input_size = X.shape[2]
model = EnhancedLSTMModel(
    input_size=input_size, 
    hidden_size=128,  # 增加隐藏层大小
    num_layers=3,     # 增加层数
    output_size=input_size, 
    dropout=0.3       # 增加dropout
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ====== 4. 训练LSTM模型 ======
# 使用AdamW优化器和学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
criterion = nn.MSELoss()

# 早停
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=5)
batch_size = 64
epochs = 50  # 增加训练轮数
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    idx = np.random.permutation(len(X_train))
    X_train_shuf, y_train_shuf = X_train[idx], y_train[idx]
    train_loss = 0
    
    # 使用梯度累积
    accumulation_steps = 4
    optimizer.zero_grad()
    
    for i in range(0, len(X_train_shuf), batch_size):
        xb = torch.tensor(X_train_shuf[i:i+batch_size], dtype=torch.float32).to(device)
        yb = torch.tensor(y_train_shuf[i:i+batch_size], dtype=torch.float32).to(device)
        
        out = model(xb)
        loss = criterion(out, yb)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i // batch_size + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * len(xb) * accumulation_steps
    
    train_loss /= len(X_train_shuf)
    
    # 验证
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_val, dtype=torch.float32).to(device)
        yb = torch.tensor(y_val, dtype=torch.float32).to(device)
        out = model(xb)
        val_loss = criterion(out, yb).item()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # 早停检查
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# ====== 5. 保存模型权重 ======
os.makedirs('model_ckpt', exist_ok=True)
torch.save(model.state_dict(), 'model_ckpt/best_lstm_multi_asset.pth')
print('Multi-asset LSTM weights saved to model_ckpt/best_lstm_multi_asset.pth')

# ====== 6. 绘制loss曲线 ======
plt.figure()
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Training Loss (Multi-Asset Multi-Factor)')
plt.savefig('model_ckpt/lstm_multi_asset_loss_curve.png')
plt.close()

# ====== 7. 验证集预测与真实值可视化（只画收盘价） ======
model.eval()
with torch.no_grad():
    xb = torch.tensor(X_val, dtype=torch.float32).to(device)
    yb = torch.tensor(y_val, dtype=torch.float32).to(device)
    pred = model(xb).cpu().numpy()
    true = yb.cpu().numpy()

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['legend.fontsize'] = 10

# 只画每只资产的收盘价预测
close_indices = [i for i, col in enumerate(feature_cols) if '收盘' in col]
fig, axes = plt.subplots(len(close_indices), 1, figsize=(12, 2*len(close_indices)), sharex=True)
if len(close_indices) == 1:
    axes = [axes]
for i, idx in enumerate(close_indices):
    axes[i].plot(pred[:200, idx], label=f'Pred {feature_cols[idx]}', linestyle='--')
    axes[i].plot(true[:200, idx], label=f'True {feature_cols[idx]}')
    axes[i].set_ylabel('Norm Close')
    axes[i].legend()
    axes[i].grid(True, linestyle='--', alpha=0.3)
axes[-1].set_xlabel('Sample')
fig.suptitle('LSTM Validation: Predicted vs True (收盘价, 前200样本)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('model_ckpt/lstm_multi_asset_val_pred_vs_true_close.png')
plt.close()
print('已美化并保存多资产LSTM验证集收盘价预测与真实值对比图到 model_ckpt/lstm_multi_asset_val_pred_vs_true_close.png') 