import os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy

# 数据加载和路径设置
data = pd.read_csv("cet_mean_station_series.csv", parse_dates=['date'])
data.set_index('date', inplace=True)

os.makedirs("Figures", exist_ok=True)
os.makedirs("Models", exist_ok=True)
save_path = "Figures"

# 原始数据预处理
raw_data = data['Pershore_College'].dropna()
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
total_sequences = len(X_raw)
train_split = int(total_sequences * 0.7)
val_split = int(total_sequences * 0.15)
X_train_raw = X_raw[:train_split]
X_val_raw = X_raw[train_split:train_split + val_split]
X_test_raw = X_raw[train_split + val_split:]
y_train_raw = y_raw[:train_split]
y_val_raw = y_raw[train_split:train_split + val_split]
y_test_raw = y_raw[train_split + val_split:]

# 初始化并拟合归一化器
X_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_raw)
y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y_train_raw)

# 保存归一化器
dump({'X': X_scaler, 'y': y_scaler}, 'Models/scalers.joblib')

# 归一化处理
X_train = X_scaler.transform(X_train_raw)
X_val = X_scaler.transform(X_val_raw)
X_test = X_scaler.transform(X_test_raw)
y_train = y_scaler.transform(y_train_raw)
y_val = y_scaler.transform(y_val_raw)
y_test = y_scaler.transform(y_test_raw)

# 转换为Tensor
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

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

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=forecast_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(-1, history_size, 1)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        output = self.fc(last_hidden)
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=forecast_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(-1, history_size, 1)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        output = self.fc(last_hidden)
        return output

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=forecast_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(-1, history_size, 1)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        output = self.fc(last_hidden)
        return output

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, embed_dim=64, num_heads=4, num_layers=3, output_size=forecast_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_size)
    
    def forward(self, x):
        x = x.view(-1, history_size, 1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        out = self.transformer_encoder(x)
        last_out = out[-1]
        output = self.fc(last_out)
        return output

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=100, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_model_wts = copy.deepcopy(model.state_dict())
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
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    model.load_state_dict(best_model_wts)
    return history

# 评估函数
def evaluate_model(model, loader, y_scaler, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            output = model(X)
            predictions.append(output.cpu().numpy())
            actuals.append(y.numpy())
    all_predictions = np.concatenate(predictions, axis=0)
    all_actuals = np.concatenate(actuals, axis=0)
    predictions_inv = y_scaler.inverse_transform(all_predictions)
    actuals_inv = y_scaler.inverse_transform(all_actuals)
    return predictions_inv, actuals_inv

# 指标计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, rmse, mae, mape

# 检查模型保存路径
MODEL_PATHS = {
    'MLP': 'Models/MLP.pth',
    'KAN': 'Models/KAN.pth',
    'RNN': 'Models/RNN.pth',
    'LSTM': 'Models/LSTM.pth',
    'GRU': 'Models/GRU.pth',
    'Transformer': 'Models/Transformer.pth'
}

# 模型训练或加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

models = {}
histories = {}

for model_name in MODEL_PATHS:
    if os.path.exists(MODEL_PATHS[model_name]):
        print(f"\n加载预训练的{model_name}模型...")
        if model_name == 'MLP':
            model = MLP().to(device)
        elif model_name == 'KAN':
            model = KAN().to(device)
        elif model_name == 'RNN':
            model = RNNModel().to(device)
        elif model_name == 'LSTM':
            model = LSTMModel().to(device)
        elif model_name == 'GRU':
            model = GRUModel().to(device)
        elif model_name == 'Transformer':
            model = TransformerModel().to(device)
        model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=device))
        model.eval()
        histories[model_name] = {'train': [], 'val': []}
    else:
        print(f"\n开始训练{model_name}模型...")
        if model_name == 'MLP':
            model = MLP().to(device)
        elif model_name == 'KAN':
            model = KAN().to(device)
        elif model_name == 'RNN':
            model = RNNModel().to(device)
        elif model_name == 'LSTM':
            model = LSTMModel().to(device)
        elif model_name == 'GRU':
            model = GRUModel().to(device)
        elif model_name == 'Transformer':
            model = TransformerModel().to(device)
        history = train_model(model, train_loader, val_loader)
        torch.save(model.state_dict(), MODEL_PATHS[model_name])
        histories[model_name] = history

    models[model_name] = model

# 可视化训练曲线
if any(len(histories[model_name]['train']) > 0 for model_name in histories):
    plt.figure(figsize=(10, 5))
    for model_name in histories:
        if histories[model_name]['train']:
            plt.plot(histories[model_name]['train'], label=f'{model_name} Train')
            plt.plot(histories[model_name]['val'], label=f'{model_name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'training_curve.png'), dpi=300)
    plt.show()
else:
    print("No training history to plot.")

# 收集指标
metrics_list = []
loaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}
for model_name in models:
    model = models[model_name]
    for dataset_name in ['train', 'val', 'test']:
        loader = loaders[dataset_name]
        preds, acts = evaluate_model(model, loader, y_scaler, device)
        mse, rmse, mae, mape = calculate_metrics(acts, preds)
        metrics_list.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

# 保存指标到CSV
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('metrics.csv', index=False)
print(metrics_df)

# 可视化预测（仅为MLP和KAN）
num_plots = 8
sample_indices = np.linspace(0, len(X_test)-1, num_plots, dtype=int)

for i, idx in enumerate(sample_indices):
    history = X_scaler.inverse_transform(X_test[idx].numpy().reshape(1, -1))[0][-history_size:]
    actual = y_scaler.inverse_transform(y_test[idx].numpy().reshape(1, -1))[0]
    mlp_pred = y_scaler.inverse_transform(models['MLP'](X_test[idx].unsqueeze(0).to(device)).detach().cpu().numpy())[0]
    kan_pred = y_scaler.inverse_transform(models['KAN'](X_test[idx].unsqueeze(0).to(device)).detach().cpu().numpy())[0]
    
    time_hist = np.arange(0, history_size)
    time_pred = np.arange(history_size, history_size+forecast_size)
    
    plt.figure(figsize=(10,3))
    plt.plot(time_hist, history, 'b-', label='History')
    plt.plot(time_pred, actual, 'g-', label='Actual')
    plt.plot(time_pred, mlp_pred, 'r--', label='MLP')
    plt.plot(time_pred, kan_pred, 'm-.', label=r'Tem$^{2}$-KAN')
    plt.axvline(history_size, color='gray', linestyle=':')
    plt.xlim(0, history_size+forecast_size)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'predictions_{i}.png'), dpi=1200)
    plt.close()