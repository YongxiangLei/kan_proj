import os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load

history_size = 1000
forecast_size = 300
n_steps = history_size + forecast_size
batch_size = 32
num_epochs = 100
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

        mlp_model_path = f"Models/{col}/MLP.pth"
        kan_model_path = f"Models/{col}/KAN.pth"

        mlp_history, kan_history = None, None

        mlp = MLP().to(device)
        if os.path.exists(mlp_model_path):
            print(f"Loading MLP model for {col}...")
            mlp.load_state_dict(torch.load(mlp_model_path, map_location=device))

            if os.path.exists(f"Models/{col}/MLP_history.csv"):
                mlp_history = pd.read_csv(f"Models/{col}/MLP_history.csv").to_dict(orient='list')
        else:
            print(f"\nTraining MLP for {col}...")
            mlp_history = train_model(mlp, train_loader, test_loader, num_epochs=num_epochs)
            if mlp_history:
                pd.DataFrame(mlp_history).to_csv(f"Models/{col}/MLP_history.csv", index=False)
            torch.save(mlp.state_dict(), mlp_model_path)

        kan = KAN().to(device)
        if os.path.exists(kan_model_path):
            print(f"Loading KAN model for {col}...")
            kan.load_state_dict(torch.load(kan_model_path, map_location=device))

            if os.path.exists(f"Models/{col}/KAN_history.csv"):
                kan_history = pd.read_csv(f"Models/{col}/KAN_history.csv").to_dict(orient='list')
        else:
            print(f"\nTraining KAN for {col}...")
            kan_history = train_model(kan, train_loader, test_loader, num_epochs=num_epochs)
            if kan_history:
                pd.DataFrame(kan_history).to_csv(f"Models/{col}/KAN_history.csv", index=False)
            torch.save(kan.state_dict(), kan_model_path)

        if mlp_history and kan_history:
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
        else:
            print("No training history found.")

        print(f"\nEvaluating all test samples for {col}...")
        mlp.eval()
        kan.eval()

        all_mlp_mse = []
        all_mlp_rmse = []
        all_mlp_mae = []
        all_mlp_mape = []

        all_kan_mse = []
        all_kan_rmse = []
        all_kan_mae = []
        all_kan_mape = []

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        num_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size != 0 else 0)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_test))

            X_batch = X_tensor[start_idx:end_idx]
            y_batch = y_tensor[start_idx:end_idx]

            with torch.no_grad():
                mlp_pred_norm = mlp(X_batch).cpu().numpy()
                kan_pred_norm = kan(X_batch).cpu().numpy()

                actual_norm = y_batch.cpu().numpy()

                for j in range(len(mlp_pred_norm)):
                    mlp_mse = mean_squared_error(actual_norm[j], mlp_pred_norm[j])
                    mlp_rmse = np.sqrt(mlp_mse)
                    mlp_mae = mean_absolute_error(actual_norm[j], mlp_pred_norm[j])
                    mlp_mape = np.mean(np.abs((actual_norm[j] - mlp_pred_norm[j]) / (actual_norm[j]+1e-8))) * 100

                    all_mlp_mse.append(mlp_mse)
                    all_mlp_rmse.append(mlp_rmse)
                    all_mlp_mae.append(mlp_mae)
                    all_mlp_mape.append(mlp_mape)

                    kan_mse = mean_squared_error(actual_norm[j], kan_pred_norm[j])
                    kan_rmse = np.sqrt(kan_mse)
                    kan_mae = mean_absolute_error(actual_norm[j], kan_pred_norm[j])
                    kan_mape = np.mean(np.abs((actual_norm[j] - kan_pred_norm[j]) / (actual_norm[j]+1e-8))) * 100

                    all_kan_mse.append(kan_mse)
                    all_kan_rmse.append(kan_rmse)
                    all_kan_mae.append(kan_mae)
                    all_kan_mape.append(kan_mape)
        avg_metrics = {
            'MLP': {'mse': np.mean(all_mlp_mse), 'rmse': np.mean(all_mlp_rmse), 'mae': np.mean(all_mlp_mae), 'mape': np.mean(all_mlp_mape)},
            'KAN': {'mse': np.mean(all_kan_mse), 'rmse': np.mean(all_kan_rmse), 'mae': np.mean(all_kan_mae), 'mape': np.mean(all_kan_mape)}
        }

        print(f"\nAverage Metrics for {col} (All {len(X_test)} test samples):")
        print(f"MLP: MSE={avg_metrics['MLP']['mse']:.4f}, RMSE={avg_metrics['MLP']['rmse']:.4f}, MAE={avg_metrics['MLP']['mae']:.4f}, MAPE={avg_metrics['MLP']['mape']:.2f}")
        print(f"KAN: MSE={avg_metrics['KAN']['mse']:.4f}, RMSE={avg_metrics['KAN']['rmse']:.4f}, MAE={avg_metrics['KAN']['mae']:.4f}, MAPE={avg_metrics['KAN']['mape']:.2f}") 
        
        pd.DataFrame([avg_metrics]).to_csv(f"Models/{col}/average_metrics_all_test_samples.csv", index=False)

        results = []
        num_plots = 5
        sample_indices = np.linspace(0, len(X_test)-1, num_plots, dtype=int)

        scaler = load(f"Models/{col}/scalers.joblib")
        X_scaler = scaler['X']
        y_scaler = scaler['y']

        for i, idx in enumerate(sample_indices):
            plt.figure(figsize=(10, 4))
            raw_history = X_scaler.inverse_transform(X_test[idx].reshape(1, -1))[0]
            actual = y_scaler.inverse_transform(y_test[idx].reshape(1, -1))[0]

            X_tensor = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0).to(device)

            mlp.eval()
            kan.eval()
            with torch.no_grad():
                mlp_pred_norm = mlp(X_tensor).cpu().numpy().flatten()
                kan_pred_norm = kan(X_tensor).cpu().numpy().flatten()

                mlp_pred = y_scaler.inverse_transform(mlp_pred_norm.reshape(1, -1)).flatten()
                kan_pred = y_scaler.inverse_transform(kan_pred_norm.reshape(1, -1)).flatten()

                # mlp_pred = y_scaler.inverse_transform(mlp(X_tensor).cpu().numpy())
                # kan_pred = y_scaler.inverse_transform(kan(X_tensor).cpu().numpy())
            
            time_hist = np.arange(0, history_size)
            time_pred = np.arange(history_size, history_size+forecast_size)

            plt.plot(time_hist, raw_history, 'b-', label='History')
            plt.plot(time_pred, actual, 'g-', label='Actual')
            plt.plot(time_pred, mlp_pred.flatten(), 'r--', label='MLP Prediction')
            plt.plot(time_pred, kan_pred.flatten(), 'm-.', label='ITKAN Prediction')
            plt.axvline(history_size, color='gray', linestyle=':', linewidth=1.2)
            plt.xlim(0, history_size+forecast_size)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time Step', fontsize=14, fontname='Times New Roman')
            plt.ylabel('Temperature (°C)', fontsize=14,fontname='Times New Roman')
            plt.legend(loc='upper left',fontsize=10,framealpha=0.8)
            plt.tight_layout()
            plt.savefig(f"Figures/{col}/prediction_comparison_{i+1}.png", bbox_inches='tight',dpi=1200)
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
        
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
    
    return history


if __name__ == '__main__':
    main_process()