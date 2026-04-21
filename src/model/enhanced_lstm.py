import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=4)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.permute(1, 0, 2)
        attn_out, _ = self.attention(out, out, out)
        out = out.permute(1, 0, 2)
        out = self.layer_norm(out + attn_out.permute(1, 0, 2))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out 