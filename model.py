import torch.nn as nn

class SleepStageModel(nn.Module):
    def __init__(self, input_dim=3000, num_classes=5, dropout_rate=0.4, nodes=128):
        super().__init__()

        self.cnn = nn.Sequential(
            ResidualBlock(in_channels=3, out_channels=nodes, kernel_size=49, stride=1, padding=25),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout_rate),
            ResidualBlock(in_channels=nodes, out_channels=nodes*2, kernel_size=25, stride=1, padding=12),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout_rate),
        )

        self.gru = nn.GRU(input_size=nodes*2, hidden_size=nodes*2, num_layers=4,
                          batch_first=True, bidirectional=True, dropout=dropout_rate)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=nodes*4, nhead=8, dim_feedforward=nodes*8,
                                       dropout=dropout_rate, batch_first=True),
            num_layers=4
        )

        self.fc = nn.Sequential(
            nn.Linear(nodes*4, nodes*2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(nodes*2, num_classes)
        )

    def forward(self, x):
        if x.dim() > 3:
            x = x.squeeze()

        x = self.cnn(x)                         # (B, 128, seq_len)
        x = x.permute(0, 2, 1)                  # (B, seq_len, 128)

        # LSTM 处理时序
        x, _ = self.gru(x)  # (batch_size, seq_len, 256)  # 双向 LSTM，hidden_size * 2

        # Transformer 处理时序
        x = self.transformer(x)  # (batch_size, seq_len, 128)

        x = x.mean(dim=1)               # (B, 256)

        return self.fc(x)

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()

        # 主分支
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding= kernel_size//2 )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 残差分支
        self.downsample = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x  # 保存输入

        # 主分支
        out = self.conv1(x)
        # print(f"conv1 output shape: {out-best.shape}")
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # print(f"conv2 output shape: {out-best.shape}")
        out = self.bn2(out)

        # 残差分支
        identity = self.downsample(identity)
        # print(f"downsample output shape: {identity.shape}")

        # 合并主分支和残差分支
        out += identity
        out = self.relu(out)
        return out
