import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=511)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh'])
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--pred_plot_len', type=int, default=200)
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
args = parser.parse_args()

# 设备
if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)
print(f'使用设备: {device}')

# 加载数据
X_train = np.load('data/train_X.npy')
y_train = np.load('data/train_y.npy')
X_val = np.load('data/val_X.npy')
y_val = np.load('data/val_y.npy')

# 转为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size)

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, activation):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.act(out)
        out = self.fc(out)
        return out

model = LSTMModel(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.dropout, args.activation).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 训练
best_val_loss = float('inf')
best_epoch = 0
train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_rmses, val_rmses = [], []
os.makedirs('model_ckpt', exist_ok=True)
for epoch in range(1, args.epochs+1):
    model.train()
    train_loss, train_mae, train_rmse = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
        train_mae += mean_absolute_error(yb.cpu().detach().numpy(), pred.cpu().detach().numpy()) * len(xb)
        train_rmse += np.sqrt(mean_squared_error(yb.cpu().detach().numpy(), pred.cpu().detach().numpy())) * len(xb)
    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)
    train_rmse /= len(train_loader.dataset)
    train_losses.append(train_loss)
    train_maes.append(train_mae)
    train_rmses.append(train_rmse)
    # 验证
    model.eval()
    val_loss, val_mae, val_rmse = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * len(xb)
            val_mae += mean_absolute_error(yb.cpu().numpy(), pred.cpu().numpy()) * len(xb)
            val_rmse += np.sqrt(mean_squared_error(yb.cpu().numpy(), pred.cpu().numpy())) * len(xb)
    val_loss /= len(val_loader.dataset)
    val_mae /= len(val_loader.dataset)
    val_rmse /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_maes.append(val_mae)
    val_rmses.append(val_rmse)
    print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_mae={train_mae:.4f}, val_mae={val_mae:.4f}, train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}')
    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), f'model_ckpt/best_lstm.pth')
        print('  [*] 保存最佳模型')
    elif epoch - best_epoch >= args.patience:
        print('  [!] 早停')
        break
    scheduler.step(val_loss)
# 保存loss曲线
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training Curve')
plt.subplot(2,1,2)
plt.plot(train_maes, label='train_mae')
plt.plot(val_maes, label='val_mae')
plt.plot(train_rmses, label='train_rmse')
plt.plot(val_rmses, label='val_rmse')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MAE / RMSE')
plt.tight_layout()
plt.savefig('model_ckpt/loss_curve.png')
plt.close()
print('已保存loss曲线到model_ckpt/loss_curve.png')

# 载入最佳模型，输出部分预测效果
model.load_state_dict(torch.load('model_ckpt/best_lstm.pth', map_location=device))
model.eval()
with torch.no_grad():
    pred = model(X_val.to(device)).squeeze().cpu().numpy()
    y_true = y_val.squeeze().cpu().numpy()
# 可视化预测效果
plot_len = min(args.pred_plot_len, len(pred))
plt.figure(figsize=(12,5))
plt.plot(pred[:plot_len], label='Pred')
plt.plot(y_true[:plot_len], label='True')
plt.legend()
plt.title(f'Val Prediction vs True (前{plot_len}样本)')
plt.savefig('model_ckpt/val_pred_vs_true.png')
plt.close()
print(f'已保存部分预测效果到model_ckpt/val_pred_vs_true.png (前{plot_len}样本)')

# 评估指标
mae = mean_absolute_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
mse = mean_squared_error(y_true, pred)
print(f'验证集最终指标: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}') 