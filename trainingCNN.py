import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def load_training_data(noisy_data_path, ideal_data_path):
    """加载带噪声的效率和理想效率数据"""
    noisy_data = np.load(noisy_data_path)
    ideal_data = np.load(ideal_data_path)
    return noisy_data, ideal_data


# 1D卷积自编码器模型
class DenoisingConvAutoencoder(nn.Module):
    def __init__(self, input_len=500):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 500)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)  # (batch, 500)


# Total Variation Loss
def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))


# 自定义损失函数（MSE + TV Loss）
def loss_fn(pred, target, alpha=1.0, beta=0.01):
    mse = nn.functional.mse_loss(pred, target)
    tv = total_variation_loss(pred)
    return alpha * mse + beta * tv


def train_autoencoder(noisy_data, ideal_data, batch_size=32, epochs=50, learning_rate=0.001):
    """训练卷积自编码器"""
    input_len = noisy_data.shape[1]
    model = DenoisingConvAutoencoder(input_len=input_len).float()

    # 转换为 Tensor
    noisy_tensor = torch.tensor(noisy_data, dtype=torch.float32)
    ideal_tensor = torch.tensor(ideal_data, dtype=torch.float32)

    dataset = TensorDataset(noisy_tensor, ideal_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model, losses


def denoise_data(model, noisy_data):
    """使用训练好的模型进行去噪"""
    model.eval()
    noisy_tensor = torch.tensor(noisy_data, dtype=torch.float32)
    with torch.no_grad():
        denoised = model(noisy_tensor)
    #return denoised.detach().cpu().numpy()
    # 如果在GPU上进行计算，确保先将数据转移到CPU
    denoised_data = denoised.detach().cpu().tolist()

    return denoised_data

def plot_loss_history(losses, epochs):
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Loss History')
    plt.legend()
    plt.show()


def plot_denoised_results(noisy_data, ideal_data, denoised_data, num_samples=5):
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.plot(noisy_data[i], color='red')
        plt.title(f"Denoisy {i + 1}")

        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.plot(ideal_data[i], color='green')
        plt.title(f"Ideal {i + 1}")

        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.plot(denoised_data[i], color='blue')
        plt.title(f"Smoothed {i + 1}")

    plt.tight_layout()
    plt.show()



from skimage.restoration import denoise_tv_chambolle

def tv_denoise(signal, weight=0.1):
    signal = np.asarray(signal)
    return denoise_tv_chambolle(signal, weight=weight)

import ruptures as rpt


def segment_mean_smoothing(signal, model="l2", penalty=3):
    signal = np.asarray(signal)
    algo = rpt.Pelt(model=model).fit(signal)
    change_points = algo.predict(pen=penalty)

    smoothed = signal.copy()
    start = 0
    for end in change_points:
        smoothed[start:end] = np.mean(signal[start:end])
        start = end
    return smoothed

from scipy.signal import medfilt


def enhanced_platform_smoother(signal, threshold1=0.1, large_jump_threshold=0.5):
    """
    piecewise constant 平滑器。
    - threshold1: 相邻段的均值变化必须超过这个才算跳变
    - large_jump_threshold: 如果某一段的波动范围超过这个阈值，视为短期大幅波动，不进行平滑
    """
    signal = np.asarray(signal)
    smoothed = np.zeros_like(signal)
    n = len(signal)
    start = 0

    while start < n:
        # 找到下一跳变点
        end = start + 1
        while end < n and abs(signal[end] - signal[start]) <= threshold1:
            end += 1

        segment = signal[start:end]
        seg_range = segment.max() - segment.min()

        if seg_range > large_jump_threshold:
            # 短期大幅波动，不平滑，直接复制原始信号
            smoothed[start:end] = segment
        else:
            # 正常平台或小幅波动，使用段均值平滑
            seg_mean = segment.mean()
            smoothed[start:end] = seg_mean

        start = end

    return smoothed


#
# def final_smooth(signal):
#     signal_tv = tv_denoise(signal, weight=0.05)
#     signal_final = segment_mean_smoothing(signal_tv, penalty=5)
#     return signal_final



def final_smooth(signal):
    return enhanced_platform_smoother(signal)


# === 主流程 ===

# 加载数据
noisy_data, ideal_data = load_training_data('new_noisy_efficiency.npy', 'new_ideal_efficiency.npy')

# 训练模型
model, losses = train_autoencoder(noisy_data, ideal_data, epochs=50)

# 去噪处理
denoised_data = denoise_data(model, noisy_data)

denoised_smoothed = np.array([final_smooth(seq) for seq in denoised_data])


# 绘图展示
#plot_loss_history(losses, epochs=50)
plot_denoised_results(denoised_data, ideal_data, denoised_smoothed)

ideal_filename='truth1.npy'
smooth_filename='smooth1.npy'

print(f"shape {ideal_data.shape}")
print(f"shape {denoised_smoothed.shape}")
np.save(ideal_filename, ideal_data)
np.save(smooth_filename, denoised_smoothed)