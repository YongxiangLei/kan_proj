import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kan import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs("Figures", exist_ok=True)
os.makedirs("Modelss", exist_ok=True)

# 数据加载与归一化
data_df = pd.read_csv("cet_mean_station_series.csv")
if 'Radcliffe' in data_df.columns:
    temp_data = data_df['Radcliffe'].values
else:
    temp_data = data_df.iloc[:, 0].values

temp_min = temp_data.min()
temp_max = temp_data.max()
temp_data_norm = (temp_data - temp_min) / (temp_max - temp_min)

# 构造数据集
def create_dataset(data, history_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - history_size - forecast_horizon + 1):
        X.append(data[i:i+history_size])
        y.append(data[i+history_size:i+history_size+forecast_horizon])
    return np.array(X), np.array(y)

def build_layers(input_size, hidden_layers, output_size):
    return [input_size] + hidden_layers + [output_size]

history_size = 1000
forecast_horizon = 300
hidden_layers = [64, 32]
X, y = create_dataset(temp_data_norm, history_size, forecast_horizon)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# 按时间顺序划分（最后20%作为测试集）
test_size = int(0.2 * len(X))
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)  
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 设置不同的 k 和 grid 组合
k_values = [3, 5, 10]
grid_values = [3, 5, 10]

plt.figure(figsize=(12, 8))

# 训练模型并绘制每个组合的损失曲线
for k in k_values:
    for grid in grid_values:
        model = KAN(width=build_layers(history_size,hidden_layers,forecast_horizon), k=k, grid=grid).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        num_epochs = 200
        train_losses = []

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if epoch % 10==0:
                print(f"{epoch} --- training loss==: {loss}")

        model_name = f"KAN_k{k}_Grid{grid}.pth"
        model_path = os.path.join("Modelss", model_name)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_name}")
        
        plt.plot(train_losses, label=f'k={k}, grid={grid}',linestyle='--')

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('MSE Loss', fontsize=18)
plt.tick_params(axis='x', labelsize=16)  
plt.tick_params(axis='y', labelsize=16)  
plt.legend()
# plt.title('KAN Training Loss for Different k and grid', fontsize=16)

plt.savefig("Figures/KAN_Training_Loss_Comparison.png", dpi=1200)



# -------------------------------
# 模型评估与预测结果展示（反归一化）
# -------------------------------
# 加载最佳模型（例如 k=5, grid=5）
model = KAN(width=build_layers(history_size,hidden_layers,forecast_horizon), k=5, grid=5).to(device)
model.load_state_dict(torch.load('Models/KAN_k5_Grid5.pth'))

model.eval()

results = []

# 在测试集上进行预测
with torch.no_grad():
    predictions_norm = model(X_test_tensor).cpu().numpy()  # 将结果移动到 CPU
    actual_norm = y_test_tensor.cpu().numpy()  # 将结果移动到 CPU
    mse = mean_squared_error(actual_norm, predictions_norm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_norm, predictions_norm)
    mape = np.mean(np.abs((actual_norm - predictions_norm) / actual_norm)) * 100  # 百分比形式
    results.append({'mse': mse, 'rmse':rmse, 'mae': mae, 'mape':mape})
    print(results)



# 反归一化：恢复到原始温度尺度
predictions = predictions_norm * (temp_max - temp_min) + temp_min
actual = actual_norm * (temp_max - temp_min) + temp_min


# 绘制测试集中前5个样本的预测结果与真实值对比图
num_samples_to_plot = 8

for i in range(num_samples_to_plot):
    plt.figure(figsize=(10,3))
    sample_idx = i  # 选择前5个样本
    plt.plot(actual[sample_idx], linestyle='-', label='Actual')
    plt.plot(predictions[sample_idx], linestyle='--', label='Predicted')
    plt.set_ylabel("Temperature (°C)", fontsize=14)
    plt.set_ylim(temp_min - 2, temp_max + 2) 
    plt.legend()
    plt.set_xlabel("Time Step",fontsize=14)
    # plt.suptitle("KAN Forecasting: Actual vs Predicted (Denormalized)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"Figures/KAN_Forecasting_{i}.png", dpi=1200, bbox_inches='tight')
    plt.close()