import os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load

# 数据加载和路径设置
data = pd.read_csv("cet_mean_station_series.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

os.makedirs("Figures", exist_ok=True)
os.makedirs("Models", exist_ok=True)
save_path = "Figures"

# 原始数据预处理
raw_data = data['Rothamsted'].dropna()
raw_data = raw_data[raw_data >= -20].values.reshape(-1, 1)

# 创建输入输出对
history_size = 1000
forecast_size = 300
n_steps = history_size + forecast_size

X_raw = []
y_raw = []
for i in range(len(raw_data) - n_steps - 1):
    X_raw.append(raw_data[i:i + history_size].flatten())
    y_raw.append(raw_data[i + history_size:i + history_size + forecast_size].flatten())

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

# 数据集分割
split_idx = int(len(X_raw) * 0.8)
X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

# 初始化并拟合归一化器
X_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_raw)
y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y_train_raw)

# 保存归一化器
dump({'X': X_scaler, 'y': y_scaler}, 'Models/scalers.joblib')

# 归一化处理
X_train = X_scaler.transform(X_train_raw)
X_test = X_scaler.transform(X_test_raw)
y_train = y_scaler.transform(y_train_raw)
y_test = y_scaler.transform(y_test_raw)

# 转换为Tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), 
                         batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), 
                        batch_size=batch_size)

# 模型定义
class MLP(nn.Module):
    def __init__(self, input_size=history_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_size)
        )

    def forward(self, x):
        return self.net(x)

class KAN(nn.Module):
    def __init__(self, input_size=history_size):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),
                nn.GELU(),
                nn.Linear(16, 8),
                nn.Tanh()
            ) for _ in range(input_size)
        ])
        self.combiner = nn.Sequential(
            nn.Linear(input_size*8, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, forecast_size)
        )

    def forward(self, x):
        features = [branch(x[:, i].unsqueeze(1)) for i, branch in enumerate(self.branches)]
        return self.combiner(torch.cat(features, dim=1))

# 训练函数
def train_model(model, train_loader, test_loader, num_epochs=100, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    best_loss = float('inf')
    no_improve = 0
    history = {'train': [], 'val': []}
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
    
    return history

# 加载归一化器
scaler_dict = load('Models/scalers.joblib')
X_scaler = scaler_dict['X']
y_scaler = scaler_dict['y']

#检查模型保存路径
MODEL_PATHS = {'MLP':'Models/MLP.pth', 'KAN':'Models/KAN.pth'}

# 模型训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 训练/加载MLP模型
if os.path.exists(MODEL_PATHS['MLP']):
    print("\n加载预训练的MLP模型...")
    mlp = MLP().to(device)
    mlp.load_state_dict(torch.load(MODEL_PATHS['MLP'], map_location=device))
    mlp.eval()
    mlp_history = {'train': [], 'val': []}  # 空历史记录
else:
    print("\n开始训练MLP模型...")
    mlp = MLP().to(device)
    mlp_history = train_model(mlp, train_loader, test_loader)
    torch.save(mlp.state_dict(), MODEL_PATHS['MLP'])

# 训练/加载KAN模型
if os.path.exists(MODEL_PATHS['KAN']):
    print("\n加载预训练的KAN模型...")
    kan = KAN().to(device)
    kan.load_state_dict(torch.load(MODEL_PATHS['KAN'], map_location=device))
    kan.eval()
    kan_history = {'train': [], 'val': []}  # 空历史记录
else:
    print("\n开始训练KAN模型...")
    kan = KAN().to(device)
    kan_history = train_model(kan, train_loader, test_loader)
    torch.save(kan.state_dict(), MODEL_PATHS['KAN'])

# 可视化训练曲线（仅当有训练历史时显示）
if mlp_history['train'] or kan_history['train']:
    plt.figure(figsize=(10, 5))
    if mlp_history['train']:
        plt.plot(mlp_history['train'], label='MLP Train')
        plt.plot(mlp_history['val'], label='MLP Val')
    if kan_history['train']:
        plt.plot(kan_history['train'], label='KAN Train')
        plt.plot(kan_history['val'], label='KAN Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'training_curve.png'), dpi=300)
    plt.show()
else:
    print("\n跳过训练曲线绘制：没有训练历史数据")

# 预测函数
def get_predictions(model, X_tensor):
    model.eval()
    with torch.no_grad():
        return model(X_tensor.to(device)).cpu().numpy()

# 获取预测结果
mlp_preds = get_predictions(mlp, X_test)
kan_preds = get_predictions(kan, X_test)
y_actual = y_test.numpy()

# 数据反归一化
mlp_preds = y_scaler.inverse_transform(mlp_preds)
kan_preds = y_scaler.inverse_transform(kan_preds)
y_actual = y_scaler.inverse_transform(y_actual)

# 数据验证
print(f"Prediction shapes - MLP: {mlp_preds.shape}, KAN: {kan_preds.shape}, Actual: {y_actual.shape}")

# 可视化对比
num_plots = 8
sample_indices = np.linspace(0, len(X_test)-1, num_plots, dtype=int)

fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))

for i, idx in enumerate(sample_indices):
    # 历史数据（最后300个点）
    history = X_scaler.inverse_transform(X_test[idx].numpy().reshape(1, -1))[0][-history_size:]
    
    # 实际值和预测值
    actual = y_actual[idx]
    mlp_pred = mlp_preds[idx]
    kan_pred = kan_preds[idx]
    
    # 时间轴
    time_hist = np.arange(0, history_size)
    time_pred = np.arange(history_size, history_size+forecast_size)
    
    # 绘图
    plt.figure(figsize=(10,3))
    plt.plot(time_hist, history, 'b-', label='History')
    plt.plot(time_pred, actual, 'g-', label='Actual')
    plt.plot(time_pred, mlp_pred, 'r--', label='MLP')
    plt.plot(time_pred, kan_pred, 'm-.', label=r'Tem$^{2}$-KAN')
    plt.axvline(history_size, color='gray', linestyle=':')
    plt.xlim(0, history_size+forecast_size)
    plt.xlabel('Time Step',fontsize=14)
    plt.ylabel('Temperature (°C)',fontsize=14)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'predictions_{i}.png'), dpi=1200)
    plt.close()
