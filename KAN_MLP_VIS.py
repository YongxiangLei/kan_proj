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

# 可视化训练曲线
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
num_plots = 5
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
    axes[i].plot(time_hist, history, 'b-', label='History')
    axes[i].plot(time_pred, actual, 'g-', label='Actual')
    axes[i].plot(time_pred, mlp_pred, 'r--', label='MLP')
    axes[i].plot(time_pred, kan_pred, 'm-.', label='KAN')
    axes[i].axvline(history_size, color='gray', linestyle=':')
    axes[i].set_xlim(0, history_size+forecast_size)
    axes[i].grid(True)
    
    if i == 0:
        axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'predictions.png'), dpi=300)
plt.show()

# ======================
# 新增可视化代码部分
# ======================

# 可视化模型参数量对比
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

mlp_params = count_parameters(mlp)
kan_params = count_parameters(kan)

plt.figure(figsize=(8, 4))
plt.bar(['MLP', 'KAN'], [mlp_params, kan_params], color=['skyblue', 'salmon'])
plt.ylabel('Parameters')
plt.title('Model Complexity Comparison')
plt.savefig(os.path.join(save_path, 'param_comparison.png'), dpi=300)
plt.show()

# 可视化KAN分支权重分布
plt.figure(figsize=(15, 6))
selected_branches = [0, 1, 99, 199]  # 选择不同位置的典型分支
for idx, bid in enumerate(selected_branches):
    # 第一层权重
    weights = kan.branches[bid][0].weight.data.cpu().numpy().flatten()
    plt.subplot(2, 4, idx+1)
    plt.hist(weights, bins=30, color='teal', alpha=0.7)
    
    # 第二层权重
    weights = kan.branches[bid][2].weight.data.cpu().numpy().flatten()
    plt.subplot(2, 4, idx+5)
    plt.hist(weights, bins=30, color='purple', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'kan_weight_dist.png'), dpi=300)
plt.show()

# 可视化分支处理过程
# 可视化分支处理过程（修正版）
sample_idx = 0  # 选择第一个测试样本
sample_input = X_test[sample_idx].unsqueeze(0).to(device)

# 数据验证：确保输入有效性
assert not torch.isnan(sample_input).any(), "输入数据包含NaN值"
print(f"输入数据范围：[{sample_input.min():.3f}, {sample_input.max():.3f}]")
print(sample_input.shape)
print(sample_input[:,0])

# 获取分支输出（完整前向过程）
branch_outputs = []
with torch.no_grad():
    for bid in range(4):  # 仅可视化前4个分支
        # 输入特征选择
        branch_input = sample_input[:, bid].unsqueeze(1)  # [batch_size, 1]
        
        # 中间层输出
        intermediate = kan.branches[bid][:2](branch_input)  # Linear + GELU
        
        # 最终分支输出
        branch_out = kan.branches[bid](branch_input)
        branch_outputs.append({
            'input': branch_input.cpu().numpy(),
            'intermediate': intermediate.cpu().numpy(),
            'output': branch_out.cpu().numpy()
        })

# 可视化处理流程（修正后）
plt.figure(figsize=(15, 10))
for bid in range(4):
    data = branch_outputs[bid]
    
    # 原始输入可视化
    plt.subplot(4, 3, bid*3+1)
    plt.plot(data['input'].flatten(), 'b-', lw=2)
    plt.ylim(-1.1, 1.1)
    #plt.title(f'Branch {bid} Input\n(Value: {data["input"][0,0]:.2f})')
    
    # 中间激活层可视化
    plt.subplot(4, 3, bid*3+2)
    plt.plot(data['intermediate'].flatten(), 'g-', lw=2)
    #plt.title(f'After GELU\n(Range: [{data["intermediate"].min():.2f}, {data["intermediate"].max():.2f}])')
    
    # 最终输出可视化
    plt.subplot(4, 3, bid*3+3)
    plt.plot(data['output'].flatten(), 'r-', lw=2)
    #plt.title(f'Branch Output\n(Range: [{data["output"].min():.2f}, {data["output"].max():.2f}])')

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'branch_processing_fixed.png'), dpi=300)
plt.show()

# 可视化特征融合层
# combiner_weights = kan.combiner[0].weight.data.cpu().numpy()
# plt.figure(figsize=(12, 8))
# plt.imshow(combiner_weights, cmap='coolwarm', aspect='auto')
# plt.colorbar()
# plt.xlabel('Input Features')
# plt.ylabel('Hidden Neurons')
# #plt.title('Combiner Layer Weight Matrix')
# plt.savefig(os.path.join(save_path, 'combiner_weights.png'), dpi=300)
# plt.show()

# ======================
# KAN模型结构可视化（添加在现有代码中）
# ======================
# def visualize_kan_structure(kan_model, save_path):
#     plt.figure(figsize=(20, 12))
#     ax = plt.gca()
#     ax.axis('off')
    
#     # 结构参数
#     branch_spacing = 2.0
#     layer_width = 5.0
#     node_radius = 0.3
#     input_node_color = '#4CAF50'  # 绿色
#     branch_color = '#2196F3'      # 蓝色
#     combiner_color = '#FF9800'    # 橙色
#     output_color = '#F44336'      # 红色
    
#     # 绘制输入层
#     input_nodes = [(0, i) for i in range(kan_model.branches.__len__())]
#     for i, (x, y) in enumerate(input_nodes):
#         circle = plt.Circle((x, y), node_radius, color=input_node_color, zorder=3)
#         ax.add_patch(circle)
#         plt.text(x, y, f'IN_{i}', ha='center', va='center', color='white')
    
#     # 绘制分支结构
#     branch_layers = []
#     for bid, branch in enumerate(kan_model.branches):
#         layer_pos = []
#         x = 1
#         for lid, layer in enumerate(branch):
#             # 绘制层节点
#             layer_type = 'Linear' if isinstance(layer, nn.Linear) else 'Activation'
#             num_nodes = layer.out_features if hasattr(layer, 'out_features') else 1
            
#             nodes = [(x + lid*layer_width, bid*branch_spacing + (n - num_nodes//2)) 
#                     for n in range(num_nodes)]
#             layer_pos.append(nodes)
            
#             # 绘制节点
#             for (nx, ny) in nodes:
#                 circle = plt.Circle((nx, ny), node_radius, 
#                                   color=branch_color if layer_type == 'Linear' else combiner_color,
#                                   zorder=3)
#                 ax.add_patch(circle)
            
#             # 绘制层连接
#             if lid > 0:
#                 for src in layer_pos[lid-1]:
#                     for dst in nodes:
#                         ax.plot([src[0], dst[0]], [src[1], dst[1]], 
#                                color='gray', linewidth=0.5, alpha=0.5, zorder=1)
        
#         # 绘制分支输出到combiner
#         combiner_start = layer_pos[-1][0][0] + 2
#         for node in layer_pos[-1]:
#             ax.plot([node[0], combiner_start], [node[1], 0], 
#                    color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
    
#     # 绘制Combiner层
#     combiner_layers = []
#     x = combiner_start
#     for lid, layer in enumerate(kan_model.combiner):
#         layer_type = 'Linear' if isinstance(layer, nn.Linear) else 'Activation'
#         num_nodes = layer.out_features if hasattr(layer, 'out_features') else 1
        
#         nodes = [(x + lid*layer_width, (n - num_nodes//2)*2) 
#                 for n in range(num_nodes)]
#         combiner_layers.append(nodes)
        
#         # 绘制节点
#         for (nx, ny) in nodes:
#             circle = plt.Circle((nx, ny), node_radius, 
#                               color=combiner_color if layer_type == 'Linear' else output_color,
#                               zorder=3)
#             ax.add_patch(circle)
        
#         # 绘制层连接
#         if lid > 0:
#             for src in combiner_layers[lid-1]:
#                 for dst in nodes:
#                     ax.plot([src[0], dst[0]], [src[1], dst[1]], 
#                            color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    
#     # 绘制输出层
#     output_nodes = combiner_layers[-1]
#     for (x, y) in output_nodes:
#         circle = plt.Circle((x + layer_width, y), node_radius, 
#                           color=output_color, zorder=3)
#         ax.add_patch(circle)
#         plt.text(x + layer_width, y, 'OUT', ha='center', va='center', color='white')
    
#     # 设置坐标范围
#     all_x = [p[0] for layer in [input_nodes] + branch_layers + combiner_layers for p in layer]
#     all_y = [p[1] for layer in [input_nodes] + branch_layers + combiner_layers for p in layer]
#     plt.xlim(min(all_x)-2, max(all_x)+5)
#     plt.ylim(min(all_y)-2, max(all_y)+2)
    
#     # 添加图例
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label='Input Layer',
#                   markerfacecolor=input_node_color, markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label='Branch Layers',
#                   markerfacecolor=branch_color, markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label='Combiner Layers',
#                   markerfacecolor=combiner_color, markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label='Output Layer',
#                   markerfacecolor=output_color, markersize=10)
#     ]
#     plt.legend(handles=legend_elements, loc='upper right')
    
#     plt.title('KAN Architecture Visualization')
#     plt.savefig(os.path.join(save_path, 'kan_architecture.png'), dpi=300, bbox_inches='tight')
#     plt.show()

# # 执行可视化
# visualize_kan_structure(kan, save_path)