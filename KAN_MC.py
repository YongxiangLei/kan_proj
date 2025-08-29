import os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load

history_size = 1000
forecast_size = 300
n_steps = history_size + forecast_size
batch_size = 32
num_epochs = 200
min_temp = -20
data_clumns = ['Radcliffe','Rothamsted','Ross_on_Wye','Stonyhurst','Malvern','Squires_Gate','Ringway','Pershore_College']

def main_process():
    # 数据加载和路径设置
    data = pd.read_csv("cet_mean_station_series.csv", parse_dates=['date'])
    data.set_index('date', inplace=True)

    os.makedirs("Figures", exist_ok=True)
    os.makedirs("Models", exist_ok=True)
    save_path = "Figures"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for col_idx, col in enumerate(data_clumns):
        # 数据归一化
        print(f"\n{'='*40}")
        print(f"Processing column [{col_idx+1}/{len(data_clumns)}]: {col}")
        print(f"{'='*40}")

        os.makedirs(f"Models/{col}", exist_ok=True)
        os.makedirs(f"Figures/{col}", exist_ok=True)

        raw_series = data[col].dropna()
        raw_series = raw_series[raw_series >= min_temp].values.reshape(-1, 1)

        X, y = [], []
        for i in range(len(raw_series) - n_steps-1):
            X.append(raw_series[i:i+history_size].flatten())
            y.append(raw_series[i+history_size:i+n_steps].flatten())
        X = np.array(X)
        y = np.array(y)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
        y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)
        dump({'X': X_scaler, 'y': y_scaler}, f"Models/{col}/scalers.joblib")

        X_train = X_scaler.transform(X_train)
        y_train = y_scaler.transform(y_train)
        X_test = X_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test)

        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)) 

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        mlp = MLP().to(device)
        print(f"\nTraining MLP for {col}...")
        mlp_history = train_model(mlp, train_loader, test_loader, num_epochs=num_epochs)
        torch.save(mlp.state_dict(), f"Models/{col}/MLP.pth")

        kan = KAN().to(device)
        print(f"\nTraining KAN for {col}...")
        kan_history = train_model(kan, train_loader, test_loader, num_epochs=num_epochs)
        torch.save(kan.state_dict(), f"Models/{col}/KAN.pth")

        plt.figure(figsize=(10, 5))
        plt.plot(mlp_history['train'], label='MLP Train')
        plt.plot(mlp_history['val'], label='MLP Val')
        plt.plot(kan_history['train'], label='KAN Train')
        plt.plot(kan_history['val'], label='KAN Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"Figures/{col}/training_curve.png", dpi=300)
        plt.close()



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
def train_model(model, train_loader, test_loader, num_epochs):
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


if __name__ == '__main__':
    main_process()