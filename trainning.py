import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# 假设你已经准备好的训练集，包含带噪声和理想的效率数据
# 假设训练数据是带噪声效率和理想效率对，存储为numpy数组

def load_training_data(noisy_data_path, ideal_data_path):
    """加载带噪声的效率和理想效率数据"""
    noisy_data = np.load(noisy_data_path)  # 假设带噪声的效率数据是一个.npy文件
    ideal_data = np.load(ideal_data_path)  # 假设理想效率数据是一个.npy文件
    return noisy_data, ideal_data


# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 输出层，恢复到输入数据的形状
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(noisy_data, ideal_data, batch_size=32, epochs=50, learning_rate=0.001):
    """训练自编码器模型"""
    input_dim = noisy_data.shape[1]
    model = Autoencoder(input_dim)
    model = model.float()  # 转换模型到float32类型

    # 转换数据为torch tensor
    noisy_data = torch.tensor(noisy_data, dtype=torch.float32)
    ideal_data = torch.tensor(ideal_data, dtype=torch.float32)

    # 创建DataLoader
    dataset = TensorDataset(noisy_data, ideal_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 存储每个epoch的损失
    losses = []

    # 训练模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算每个epoch的平均损失并记录
        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model, losses


def denoise_data(model, noisy_data):
    """使用训练好的模型对带噪声的数据进行去噪"""
    model.eval()
    noisy_data_tensor = torch.tensor(noisy_data, dtype=torch.float32)
    with torch.no_grad():
        denoised_data = model(noisy_data_tensor)

    # 如果在GPU上进行计算，确保先将数据转移到CPU
    denoised_data = denoised_data.detach().cpu().tolist()

    return denoised_data


def plot_loss_history(losses, epochs):
    """绘制训练过程中的损失曲线"""
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Autoencoder Loss History')
    plt.show()


def plot_denoised_results(noisy_data, ideal_data, denoised_data, num_samples=5):
    """展示去噪效果"""
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.plot(noisy_data[i], label='Noisy Efficiency', color='red')
        plt.title(f"Noisy {i + 1}")

        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.plot(ideal_data[i], label='Ideal Efficiency', color='green')
        plt.title(f"Ideal {i + 1}")

        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.plot(denoised_data[i], label='Denoised Efficiency', color='blue')
        plt.title(f"Denoised {i + 1}")

    plt.tight_layout()
    plt.show()




# 假设你已经有带噪声和理想的效率数据
noisy_data, ideal_data = load_training_data('noisy_efficiency.npy', 'ideal_efficiency.npy')

# 训练自编码器并获取训练损失
model, losses = train_autoencoder(noisy_data, ideal_data, epochs=50)

# 使用训练好的自编码器进行去噪
denoised_data = denoise_data(model, noisy_data)



# 绘制训练损失曲线
plot_loss_history(losses, epochs=50)

# 绘制去噪效果对比
plot_denoised_results(noisy_data, ideal_data, denoised_data)
