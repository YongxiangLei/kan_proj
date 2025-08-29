##### Author: Yongxiang Lei
##### Date: 2025-03
##### Function: KAN model for multi-step time series forecasting
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
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

rcParams.update({
                  'font.family': 'serif',
                  'font.serif': 'Times New Roman',
                  'font.size': 22, 
                  'axes.labelsize':14,
                  'axes.titlesize':14,
                  'xtick.labelsize':18,
                  'ytick.labelsize':18,
                  'legend.fontsize':18,
                  'lines.linewidth':1.5,
                  'axes.linewidth':1.2,
                  'grid.linewidth':0.8,
                  'savefig.dpi':1200,
                  'savefig.bbox':'tight',
                  'savefig.format':'png'})

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
    # 为可视化创建新的文件夹
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
        os.makedirs(f"Figures/{col}/KAN_Features", exist_ok=True)  # KAN特征可视化文件夹

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
            plt.figure(figsize=(10, 5),tight_layout=True)
            plt.plot(mlp_history['train'], label='MLP Train')
            plt.plot(mlp_history['val'], label='MLP Val')
            plt.plot(kan_history['train'], label='KAN Train')
            plt.plot(kan_history['val'], label='KAN Val')
            plt.xlabel('Epoch',fontsize=14, fontname='Times New Roman')
            plt.ylabel('Loss',fontsize=14, fontname='Times New Roman')
            plt.legend()
            plt.savefig(f"Figures/{col}/training_curve.pdf", dpi=1200, bbox_inches='tight',transparent=True)
            plt.close()
        else:
            print("No training history found.")

        # 可视化KAN隐层特征和样条函数
        print(f"\nVisualizing KAN hidden features for {col}...")
        visualize_kan_features(kan, X_test, col, device)
            
        # 评估所有测试样本并计算平均指标
        print(f"\nEvaluating ALL test samples for {col}...")
        mlp.eval()
        kan.eval()
        mlp_params = count_parameters(mlp)
        kan_params = count_parameters(kan)
        print(f"MLP Parameters: {mlp_params:,}")
        print(f"KAN Parameters: {kan_params:,}")
        
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
        
        # 分批处理以避免内存问题
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
                
                # 计算归一化空间中的指标
                for j in range(len(mlp_pred_norm)):
                    # MLP 指标
                    mlp_mse = mean_squared_error(actual_norm[j], mlp_pred_norm[j])
                    mlp_rmse = np.sqrt(mlp_mse)
                    mlp_mae = mean_absolute_error(actual_norm[j], mlp_pred_norm[j])
                    mlp_mape = np.mean(np.abs((actual_norm[j] - mlp_pred_norm[j]) / (actual_norm[j] + 1e-8))) * 100
                    
                    all_mlp_mse.append(mlp_mse)
                    all_mlp_rmse.append(mlp_rmse)
                    all_mlp_mae.append(mlp_mae)
                    all_mlp_mape.append(mlp_mape)
                    
                    # KAN 指标
                    kan_mse = mean_squared_error(actual_norm[j], kan_pred_norm[j])
                    kan_rmse = np.sqrt(kan_mse)
                    kan_mae = mean_absolute_error(actual_norm[j], kan_pred_norm[j])
                    kan_mape = np.mean(np.abs((actual_norm[j] - kan_pred_norm[j]) / (actual_norm[j] + 1e-8))) * 100
                    
                    all_kan_mse.append(kan_mse)
                    all_kan_rmse.append(kan_rmse)
                    all_kan_mae.append(kan_mae)
                    all_kan_mape.append(kan_mape)
        
        # 计算平均指标
        avg_metrics = {
            'MLP': {
                'parameters': mlp_params,
                'mse': np.mean(all_mlp_mse),
                'rmse': np.mean(all_mlp_rmse),
                'mae': np.mean(all_mlp_mae),
                'mape': np.mean(all_mlp_mape)
            },
            'KAN': {
                'parameters': kan_params,
                'mse': np.mean(all_kan_mse),
                'rmse': np.mean(all_kan_rmse),
                'mae': np.mean(all_kan_mae),
                'mape': np.mean(all_kan_mape)
            }
        }
        
        # 保存平均指标
        print(f"\nAverage Metrics for {col} (All {len(X_test)} test samples):")
        print(f"MLP: MSE={avg_metrics['MLP']['mse']:.4f}, RMSE={avg_metrics['MLP']['rmse']:.4f}, MAE={avg_metrics['MLP']['mae']:.4f}, MAPE={avg_metrics['MLP']['mape']:.2f}, Parameters={avg_metrics['MLP']['parameters']:,}")
        print(f"KAN: MSE={avg_metrics['KAN']['mse']:.4f}, RMSE={avg_metrics['KAN']['rmse']:.4f}, MAE={avg_metrics['KAN']['mae']:.4f}, MAPE={avg_metrics['KAN']['mape']:.2f}, Parameters={avg_metrics['KAN']['parameters']:,}")
        
        # 保存平均指标到CSV
        pd.DataFrame([avg_metrics]).to_csv(f"Models/{col}/average_metrics_all_test_samples.csv", index=False)
        
        # 仍然绘制前3个样本的图表用于可视化
        num_plots = 3
        sample_indices = np.linspace(0, len(X_test)-1, num_plots, dtype=int)
        
        scaler = load(f"Models/{col}/scalers.joblib")
        X_scaler = scaler['X']
        y_scaler = scaler['y']
        
        for i, idx in enumerate(sample_indices):
            plt.figure(figsize=(10, 4))
            raw_history = X_scaler.inverse_transform(X_test[idx].reshape(1, -1))[0]
            actual = y_scaler.inverse_transform(y_test[idx].reshape(1, -1))[0]

            X_tensor = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                mlp_pred_norm = mlp(X_tensor).cpu().numpy().flatten()
                kan_pred_norm = kan(X_tensor).cpu().numpy().flatten()

                mlp_pred = y_scaler.inverse_transform(mlp_pred_norm.reshape(1, -1)).flatten()
                kan_pred = y_scaler.inverse_transform(kan_pred_norm.reshape(1, -1)).flatten()
            
            time_hist = np.arange(0, history_size)
            time_pred = np.arange(history_size, history_size+forecast_size)

            plt.plot(time_hist, raw_history, 'b-', label='History')
            plt.plot(time_pred, actual, 'g-', label='Actual')
            plt.plot(time_pred, mlp_pred.flatten(), 'r--', label='MLP Prediction')
            plt.plot(time_pred, kan_pred.flatten(), 'm-.', label='Temp$^2$-KAN Prediction')
            plt.axvline(history_size, color='gray', linestyle=':', linewidth=1.2)
            plt.xlim(0, history_size+forecast_size)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time Step', fontsize=22)
            plt.ylabel('Temperature (°C)', fontsize=22)
            plt.legend(loc='upper left', fontsize=10, framealpha=0.8)
            plt.tight_layout()
            plt.savefig(f"Figures/{col}/prediction_comparison_{i+1}.pdf", bbox_inches='tight', dpi=1200, transparent=True)
            plt.close()


# 添加KAN特征可视化函数
def visualize_kan_features(model, X_test, dataset_name, device, num_samples=3, resolution=100):
    """
    可视化KAN模型的隐层特征和样条函数
    
    参数:
    - model: 训练好的KAN模型
    - X_test: 测试数据
    - dataset_name: 数据集名称，用于保存文件
    - device: 计算设备（CPU或GPU）
    - num_samples: 要可视化的样本数量
    - resolution: 样条函数的分辨率
    """
    model.eval()
    
    # 选择一些代表性样本
    sample_indices = np.linspace(0, len(X_test)-1, num_samples, dtype=int)
    
    for sample_idx, idx in enumerate(sample_indices):
        print(f"Visualizing sample {sample_idx+1}/{num_samples}")
        
        # 获取样本数据
        X_sample = X_test[idx]
        X_tensor = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 1. 可视化输入数据
        plt.figure(figsize=(12, 6))
        plt.plot(X_sample, 'b-')
        plt.xlim(0, len(X_sample))
        # plt.title(f"Input Sequence - {dataset_name} Sample {sample_idx+1}")
        plt.xlabel("Time Step",fontname='Times New Roman',fontsize=18)
        plt.ylabel("Normalized Temperature",fontname='Times New Roman',fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"Figures/{dataset_name}/KAN_Features/input_sample_{sample_idx+1}.pdf", dpi=1200, bbox_inches='tight', transparent=True)
        plt.close()
        
        # 2. 可视化每个分支的样条函数（每个分支处理一个输入特征）
        # 创建一个范围在[-1, 1]之间的均匀分布用于评估样条函数
        x_range = torch.linspace(-1, 1, resolution).to(device)
        
        # 选择一部分分支进行可视化，以避免图表过于拥挤
        num_branches_to_show = min(10, len(model.branches))
        branch_indices = np.linspace(0, len(model.branches)-1, num_branches_to_show, dtype=int)
        
        # 可视化样条函数
        plt.figure(figsize=(15, 12))
        
        # 用不同颜色绘制每个分支的样条函数
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i/num_branches_to_show) for i in range(num_branches_to_show)]
        
        legend_entries = []
        
        for i, branch_idx in enumerate(branch_indices):
            branch = model.branches[branch_idx]
            
            # 评估样条函数在整个输入范围上的输出
            with torch.no_grad():
                spline_outputs = branch(x_range.unsqueeze(1)).cpu().numpy()
            
            # 绘制样条函数
            plt.plot(x_range.cpu().numpy(), spline_outputs, color=colors[i], linewidth=1.5)
            legend_entries.append(f"Branch {branch_idx}")
            
            # 标记实际输入点在样条函数上的位置
            # with torch.no_grad():
            #     actual_output = branch(X_tensor[:, branch_idx].unsqueeze(1)).cpu().numpy()[0]
            
            # plt.scatter(X_sample[branch_idx], actual_output, color=colors[i], s=50, edgecolor='black')
        
        # plt.title(f"KAN Spline Functions - {dataset_name} Sample {sample_idx+1}", fontsize=16)
        plt.xlabel("Input Value", fontsize=18,fontname='Times New Roman')
        plt.ylabel("Spline Output", fontsize=18,fontname='Times New Roman')
        plt.grid(True, alpha=0.3)
        plt.xlim(-1.1, 1.1)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # 添加图例到图表外侧
        #plt.legend(legend_entries,fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"Figures/{dataset_name}/KAN_Features/spline_functions_{sample_idx+1}.pdf", dpi=1200, bbox_inches='tight',transparent=True)
        plt.close()
        
        # 3. 可视化每个分支对于样本的激活值热力图
        with torch.no_grad():
            # 保存每个分支的输出
            branch_outputs = torch.zeros(len(model.branches), 8).to(device)
            for i, branch in enumerate(model.branches):
                branch_outputs[i] = branch(X_tensor[:, i].unsqueeze(1))
        
        # 获取分支输出的CPU版本
        branch_outputs_np = branch_outputs.cpu().numpy()
        
        # 3. 3D柱状图可视化激活模式
        fig = plt.figure(figsize=(12, 10), dpi=1200)
        ax = fig.add_subplot(111, projection='3d')

        # 数据准备
        branch_indices, feature_indices = np.meshgrid(
            np.arange(branch_outputs_np.shape[0]), 
            np.arange(branch_outputs_np.shape[1])
        )
        xpos = feature_indices.ravel()  # X轴: 特征维度
        ypos = branch_indices.ravel()   # Y轴: 分支索引
        zpos = np.zeros_like(xpos)      # 柱体基底Z值

        # 激活值参数化
        activations = branch_outputs_np.ravel()
        dx = dy = 0.8  # 柱体宽度/深度
        dz = np.abs(activations)        # 柱体高度
        colors = np.where(activations > 0, '#B2182B', '#2166AC')  # 红/蓝区分正负

        # 3D柱状图绘制
        ax.bar3d(
            xpos, ypos, zpos, 
            dx, dy, dz,
            color=colors,
            edgecolor='black',
            linewidth=0.1,
            alpha=0.8,
            zsort='average'
        )

        # 坐标轴优化
        ax.set_xlabel('Feature Dimension', labelpad=15, fontname='Times New Roman', fontsize=22)
        ax.set_ylabel('Branch Index', labelpad=15, fontname='Times New Roman', fontsize=22)
        ax.set_zlabel('Activation', labelpad=10, fontname='Times New Roman', fontsize=22)

        # 刻度间隔设置
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.zaxis.set_major_locator(plt.MaxNLocator(5))

        # 视角调整 (方位角，仰角)
        ax.view_init(elev=25, azim=-45)

        # 设置坐标轴字体大小（X, Y, Z刻度）
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(22)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(22)
        for tick in ax.zaxis.get_ticklabels():
            tick.set_fontsize(22)


        # 添加颜色映射说明
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#B2182B', edgecolor='black', label='Positive Activation'),
            Patch(facecolor='#2166AC', edgecolor='black', label='Negative Activation')
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(0.75, 0.85),
            frameon=True,
            fontsize=22,
            facecolor='white'
        )

        # 保存矢量图
        plt.savefig(
            f"Figures/{dataset_name}/KAN_Features/3d_activations_{sample_idx+1}.pdf", 
            bbox_inches='tight', 
            transparent=True
        )
        plt.close()
        
        # 4. 可视化组合后的特征（最终层之前的特征）
        with torch.no_grad():
            # 获取每个分支的输出并连接
            features = [branch(X_tensor[:, i].unsqueeze(1)) for i, branch in enumerate(model.branches)]
            combined_features = torch.cat(features, dim=1)
            
            # 应用组合器层的第一层以获取组合后的特征表示
            combined_representation = model.combiner[0](combined_features)
            combined_representation = model.combiner[1](combined_representation)  # 应用GELU激活
        
        # 将组合后的特征转换为NumPy数组进行可视化
        combined_features_np = combined_features.cpu().numpy()[0]
        combined_representation_np = combined_representation.cpu().numpy()[0]
        
        # 可视化连接特征（太长，绘制前100个）
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        show_features = min(1000, combined_features_np.shape[0])
        plt.plot(combined_features_np[:show_features])
        plt.xlim(0, show_features)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        # plt.title(f"KAN Concatenated Features (first {show_features}) - {dataset_name} Sample {sample_idx+1}")
        # plt.xlabel("Feature Index",fontname='Times New Roman',fontsize=18)
        plt.ylabel("Activation",fontname='Times New Roman',fontsize=22)
        plt.grid(True, alpha=0.3)
        
        # 可视化组合后的特征
        plt.subplot(2, 1, 2)
        plt.plot(combined_representation_np)
        # plt.title(f"KAN Combined Representation - {dataset_name} Sample {sample_idx+1}")
        plt.xlabel("Feature Index",fontname='Times New Roman',fontsize=22)
        plt.ylabel("Activation",fontname='Times New Roman',fontsize=22)
        plt.xlim(0, len(combined_representation_np))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"Figures/{dataset_name}/KAN_Features/combined_features_{sample_idx+1}.pdf", dpi=1200, transparent=True)
        plt.close()
        
        # 5. 可视化重要分支的动态范围（按激活幅度排序）
        with torch.no_grad():
            # 计算每个分支输出的范围/标准差
            branch_range = torch.zeros(len(model.branches)).to(device)
            for i, branch in enumerate(model.branches):
                # 使用全分辨率x范围评估分支
                full_range_output = branch(x_range.unsqueeze(1))
                branch_range[i] = torch.max(full_range_output) - torch.min(full_range_output)
        
        # 获取CPU版本
        branch_range_np = branch_range.cpu().numpy()
        
        # 按范围排序的索引
        sorted_indices = np.argsort(branch_range_np)[::-1]  # 降序
        
        # 选择范围最大的前10个分支
        top_branches = sorted_indices[:10]
        
        # 绘制重要分支的样条函数
        plt.figure(figsize=(15, 15))

        # 用不同颜色绘制每个重要分支的样条函数
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i / 10) for i in range(10)]  # 固定为10个颜色

        legend_entries = []

        for i, branch_idx in enumerate(top_branches):
            branch = model.branches[branch_idx]

            # 评估样条函数在整个输入范围上的输出
            with torch.no_grad():
                spline_outputs = branch(x_range.unsqueeze(1)).cpu().numpy()

            # 仅绘制这10个分支
            plt.plot(x_range.cpu().numpy(), spline_outputs, color=colors[i], linewidth=2.0)
            legend_entries.append(f"Branch {branch_idx} (Range: {branch_range_np[branch_idx]:.3f})")

            
            # 标记实际输入点在样条函数上的位置
            # with torch.no_grad():
            #     actual_output = branch(X_tensor[:, branch_idx].unsqueeze(1)).cpu().numpy()[0]
            
            # plt.scatter(X_sample[branch_idx], actual_output, color=colors[i], s=80, edgecolor='black')
        
        # plt.title(f"Top 10 Most Important KAN Spline Functions - {dataset_name} Sample {sample_idx+1}", fontsize=16)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel("Input Value", fontsize=22)
        plt.ylabel("Spline Output", fontsize=22)
        # 设置坐标轴刻度字体
        plt.tick_params(axis='both', which='major', labelsize=22)
        for label in plt.gca().get_xticklabels():
            label.set_fontname('Times New Roman')
        for label in plt.gca().get_yticklabels():
            label.set_fontname('Times New Roman')
        plt.grid(True, alpha=0.3)
        plt.xlim(-1.1, 1.1)
        
        # 添加图例
        plt.legend(legend_entries, fontsize=22)
        
        plt.tight_layout()
        plt.savefig(f"Figures/{dataset_name}/KAN_Features/important_splines_{sample_idx+1}.pdf", dpi=1200, bbox_inches='tight',transparent=True)
        plt.close()
    
    print(f"KAN visualization completed for {dataset_name}")


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


#计算模型参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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