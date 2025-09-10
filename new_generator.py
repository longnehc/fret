import numpy as np
import os
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from scipy.linalg import null_space

class IntensityandEfficiencySimulator:
    def __init__(self, total_frames, stimulate_frame, total_intensity, background_a,background_d, Xd):
        self.total_frame = total_frames
        self.stimulate_frame = stimulate_frame
        self.total_intensity = total_intensity
        self.background_a = background_a
        self.background_d = background_d
        self.Xd = Xd

        self.E1 = np.array([0.2, 0.4, 0.6, 0.8])
        self.E2 = np.array([0.3, 0.5, 0.9])

        self.T1 = np.array([
            [0.9, 0.05, 0.0, 0.05],
            [0.05, 0.8, 0.15, 0.0],
            [0.0, 0.15, 0.8, 0.05],
            [0.05, 0.0, 0.05, 0.9]
        ])
        self.T2 = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])

        self.jump_probability = {0.2:0.1, 0.8:0.05}

    def stimulator(self):
        # ------- E1 区域：初始化状态为平稳分布中最大概率对应的状态 -------
        n1 = len(self.E1)
        I1 = np.eye(n1)
        A1 = self.T1.T - I1
        Pi1 = null_space(A1)[:, 0]
        Pi1 = Pi1 / np.sum(Pi1)
        StartIndex1 = np.argmax(Pi1)
        state = self.E1[StartIndex1]

        # ------- E2 区域 -------
        #E2区域不从稳态分布中抽取初始态
        n2 = len(self.E2)

        # ------- 初始化状态记录 -------
        in_state_1 = True
        history = [state]  # 记录状态值
        state_history = [1]  # 记录属于哪一类状态（1 或 2）
        donor_seq = [] #donor通道理想光强
        acceptor_seq = [] #Acceptor通道理想光强

        kmol = self.total_intensity
        kabg = self.background_a
        kdbg = self.background_d

        # 计算初始光强（包含串扰）
        donor_initial = (1 - state) * kmol + kdbg
        acceptor_initial = state * kmol + kabg + 0.1 * (donor_initial - kdbg)  # 添加0.1倍donor串扰

        donor_seq.append(donor_initial)
        acceptor_seq.append(acceptor_initial)

        # ------- 状态模拟循环 -------
        for t in range(1, self.total_frame):
            if in_state_1:
                # 使用 T1 转移状态
                state_index = np.where(self.E1 == state)[0][0]
                next_state_probability = self.T1[state_index]
                next_state_index = np.random.choice(n1, p=next_state_probability)
                state = self.E1[next_state_index]

                if t >= self.stimulate_frame:
                    if state == 0.2 and np.random.rand() <= self.jump_probability[0.2]:
                        state = np.random.choice(self.E2)
                        in_state_1 = False
                        state_history.append(2)
                    elif state == 0.8 and np.random.rand() <= self.jump_probability[0.8]:
                        state = np.random.choice(self.E2)
                        in_state_1 = False
                        state_history.append(2)
                    else:
                        state_history.append(1)
                else:
                    state_history.append(1)
            else:
                # 进入 E2 区域后按照 T2 转移矩阵进行状态转移
                state_index = np.where(self.E2 == state)[0][0]
                next_state_probability = self.T2[state_index]
                next_state_index = np.random.choice(n2, p=next_state_probability)
                state = self.E2[next_state_index]
                state_history.append(2)

             # 计算当前帧的光强（包含串扰）
            donor_current = (1 - state) * kmol + kdbg
            acceptor_current = state * kmol + kabg + 0.1 * (donor_current - kdbg)  # 添加0.1倍donor串扰

            history.append(state)
            donor_seq.append(donor_current)
            acceptor_seq.append(acceptor_current)

        return (
        np.array(history),  # FRET 效率序列
        np.array(state_history),  # 状态分类序列（1或2）
        np.array(donor_seq),  # Donor 通道理想光强
        np.array(acceptor_seq)  # Acceptor 通道理想光强
        )

    def add_noise_crosstalk_background(self, donor_seq, acceptor_seq):
            # ---------- 1. 对 donor_seq 和 acceptor_seq 添加噪声 ----------
            donor_noisy = []
            acceptor_noisy = []
            # fret_eff = []

            for d_ideal, a_ideal in zip(donor_seq, acceptor_seq):
                # #以下注释掉的是用放大缩小法来确定增加目标snr的噪声
                # #1.设置snr
                #
                # snr=9
                #
                # # 2. 总强度
                # total_signal = d_ideal + a_ideal
                #
                # # 3. 缩放因子（基于 SNR）
                # scale_factor = snr ** 2 / total_signal
                #
                # scaled_d = d_ideal * scale_factor
                # scaled_a = a_ideal * scale_factor
                #
                # # 4. 加入泊松噪声 + 还原
                # noisy_d = np.random.poisson(scaled_d) / scale_factor
                # noisy_a = np.random.poisson(scaled_a) / scale_factor

                noisy_d = np.random.poisson(d_ideal)
                noisy_a = np.random.poisson(a_ideal)

                donor_noisy.append(noisy_d)
                acceptor_noisy.append(noisy_a)

                # # 5. FRET efficiency（如果要用）
                # eff = noisy_a / (noisy_a + noisy_d)
                # fret_eff.append(eff)


            # ---------- 2. 串扰序列 ----------
            donor_crosstalk_photons = self.total_intensity + self.background_d
            donor_crosstalk_seq = np.random.poisson(donor_crosstalk_photons, 150)

            acceptor_crosstalk_photons = self.Xd * self.total_intensity + self.background_a
            acceptor_crosstalk_seq = np.random.poisson(acceptor_crosstalk_photons, 150)

            # ---------- 3. 背景序列 ----------
            donor_background_seq = np.random.poisson(self.background_d, 150)
            acceptor_background_seq = np.random.poisson(self.background_a, 150)

            #print(donor_background_seq)
            #print(donor_noisy)

            # ---------- 4. 计算 beta 和 Xd ----------
            s1 = donor_noisy
            s2 = acceptor_noisy

            bkga = np.mean(acceptor_background_seq)
            bkgd = np.mean(donor_background_seq)
            Ia = np.mean(acceptor_noisy)
            Id = np.mean(donor_noisy)
            IbetaD = np.mean(donor_crosstalk_seq)
            Ica = np.mean(acceptor_crosstalk_seq)

            Xd = (Ica - bkga) / (IbetaD - bkgd)
            P = (IbetaD - Id) / (Ia - Ica)
            Ia0 = ((IbetaD - bkgd) + Xd * (IbetaD - bkgd) * P) / P
            IbetaA = Ia0 + bkga
            betaD = IbetaD / bkgd
            betaA = IbetaA / (bkga + Xd * (IbetaD - bkgd))

            E = []
            for i in range(0, len(donor_seq)):
                na = s2[i]
                nd = s1[i]
                Ei = (IbetaD * na - IbetaA * nd * 1 / betaA) / (IbetaD * na * (1 - 1 / betaD) + IbetaA * nd * (1 - 1 / betaA))
                E.append(Ei)
            #舍弃跳变过大的数据
            dE = np.abs(np.diff(E))
            dE = np.sort(dE)
            sigma = dE[int(round(0.682 * len(dE)))] / np.sqrt(2)  # 计算噪声水平
            print(sigma)

            if sigma >= 0.25:  # 如果噪声水平太高，丢弃这个数据
                print('This data fluctuates too much, discard')
            return np.array(E), np.array(donor_noisy),np.array(acceptor_noisy)
    #保存ground truth
    def save_to_txt(self, noisyA, noisyD, reale, ideale, filename="test.txt"):
        with open(filename, 'w') as f:
            for a, d, e, h in zip(noisyA, noisyD, reale, ideale):
                f.write(f"{h} {e} {d} {a}\n")  # 第一列到第撕裂分别为：理想FRET, 实际FRET，实际donor光强，实际acceptor光强
        print(f"Data saved to {filename}")
# #测试用单张绘图及运行
    # def plot_simulation_results(self,fret_eff, state_seq, donor_intensity, acceptor_intensity):
    #     frames = np.arange(len(fret_eff))  # x轴：帧编号
    #
    #     fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    #
    #     # 1. Donor intensity
    #     axs[0].plot(frames, donor_intensity, color='blue', label='Donor Intensity')
    #     axs[0].set_ylabel('Donor')
    #     axs[0].legend(loc='upper right')
    #     axs[0].grid(True)
    #
    #     # 2. Acceptor intensity
    #     axs[1].plot(frames, acceptor_intensity, color='red', label='Acceptor Intensity')
    #     axs[1].set_ylabel('Acceptor')
    #     axs[1].legend(loc='upper right')
    #     axs[1].grid(True)
    #
    #     # 3. FRET Efficiency
    #     axs[2].plot(frames, fret_eff, color='green', label='FRET Efficiency')
    #     axs[2].set_ylabel('Efficiency')
    #     axs[2].set_ylim(0, 1)
    #     axs[2].legend(loc='upper right')
    #     axs[2].grid(True)
    #
    #     # 4. State (1 or 2)
    #     axs[3].step(frames, state_seq, where='post', color='black', label='State (1 or 2)')
    #     axs[3].set_ylabel('State')
    #     axs[3].set_xlabel('Frame')
    #     axs[3].set_yticks([1, 2])
    #     axs[3].legend(loc='upper right')
    #     axs[3].grid(True)
    #
    #     # 调整子图间距
    #     plt.tight_layout()
    #     plt.show()
# # 创建实例并运行模拟
# simulator = IntensityandEfficiencySimulator(
#     total_frames=300,      # 总帧数
#     stimulate_frame=150,    # 刺激开始帧
#     total_intensity=85.8,  # 总强度,总强度与SNR直接挂钩，SNR=总强度/根号下（总强度+背景），SNR=3,5,7,9分别对应总强度12.6，29.3，53.6，85.8
#     background_a=5,       # 受体背景
#     background_d=5,       # 供体背景
#     Xd=0.1                 # 其他参数
# )
#
# # 运行模拟
# fret_eff, states, donor_I, acceptor_I = simulator.stimulator()
#
# # 绘制结果
# simulator.plot_simulation_results(fret_eff, states, donor_I, acceptor_I)
#
# real_e, real_d, real_a = simulator.add_noise_crosstalk_background(donor_I,acceptor_I)
#
# simulator.plot_simulation_results(real_e,states, real_d, real_a)
def generate_random_parameters():
        """Generate random parameters for IntensityandEfficiencySimulator"""
        # total_frames = np.random.randint(300, 1001)
        total_frames = 500
        stimulate_frame = total_frames // 2
        total_intensity = np.random.randint(12.6, 85.5)
        background = 5
        Xd = 0.1
        return (total_frames, stimulate_frame, total_intensity, background, Xd)
def generate_multiple_trajectories(n):
    import os
    from tkinter import Tk, filedialog

    # 选择文件夹
    # root = Tk()
    # root.withdraw()
    # save_dir = filedialog.askdirectory(title="请选择保存txt文件的文件夹")
    # root.destroy()
    #
    # if not save_dir:
    #     print("未选择文件夹，操作取消。")
    #     return

    changepoints = []  # 保存所有变换帧信息
    ideal_data_list = []
    noisy_data_list = []
    ideal_filename = 'new_ideal_efficiency.npy'
    noisy_filename = 'new_noisy_efficiency.npy'

    for i in range(1, n + 1):
        print(f"\nGenerating trajectory {i}...")

        # 1. 随机参数
        total_frames, stimulate_frame, total_intensity, background, Xd = generate_random_parameters()

        # 2. 创建模拟器
        simulator = IntensityandEfficiencySimulator(
            total_frames=total_frames,
            stimulate_frame=stimulate_frame,
            total_intensity=total_intensity,
            background_a=background,
            background_d=background,
            Xd=Xd
        )

        # 3. 模拟轨迹
        ideal_e, states, donor_I, acceptor_I = simulator.stimulator()

        # 4. 噪声 & 串扰
        noisy_e, noisy_d, noisy_a = simulator.add_noise_crosstalk_background(donor_I, acceptor_I)

        if len(noisy_e) != len(ideal_e):
            print(f"Skipping trajectory {i} due to excessive fluctuation.")
            continue

        # 5. 保存每条轨迹
        #filename = os.path.join(save_dir, f"test_{i}.txt")
        #simulator.save_to_txt(noisy_a, noisy_d, noisy_e, ideal_e, filename)

        ideal_data_list.append(ideal_e)
        noisy_data_list.append(noisy_e)

        # 6. 查找从状态1 → 2的第一次转变帧
        changepoint_frame = 0  # 默认没有变化
        for idx in range(1, len(states)):
            if states[idx - 1] == 1 and states[idx] == 2:
                changepoint_frame = idx  # 第一次从1跳到2的帧编号
                break

        changepoints.append((i, changepoint_frame))

        # 仅绘制第一个数据的图像
        if i == 1:
            plot_first_trajectory(
                donor_I=noisy_d,         # ✅ 改为加噪声后的 donor
                acceptor_I=noisy_a,      # ✅ 改为加噪声后的 acceptor
                ideal_E=ideal_e,         # 理想 FRET
                noisy_E=noisy_e,         # 实际 FRET
                stimulate_frame=stimulate_frame,
                changepoint_frame=changepoint_frame,
                total_frames=total_frames,
                total_intensity=total_intensity,
                background=background,
                Xd=Xd
            )


    ideal_data_array = np.array(ideal_data_list)
    noisy_data_array = np.array(noisy_data_list)
    print(f"shape {ideal_data_array.shape}")
    print(f"shape {noisy_data_array.shape}")

    # Save the data to .npy files
    np.save(ideal_filename, ideal_data_array)
    np.save(noisy_filename, noisy_data_array)


    # 7. 保存 changepoint.txt
    # changepoint_file = os.path.join(save_dir, "changepoint.txt")
    # with open(changepoint_file, "w") as f:
    #     for index, frame in changepoints:
    #         f.write(f"{index} {frame}\n")
    #
    # print(f"\nAll trajectories and changepoints saved to {save_dir}")

def plot_first_trajectory(
    donor_I, acceptor_I, ideal_E, noisy_E, stimulate_frame, changepoint_frame,
    total_frames, total_intensity, background, Xd
):
    import matplotlib.pyplot as plt

    frames = np.arange(len(donor_I))

    # 图 1：Donor 和 Acceptor 光强
    plt.figure(figsize=(10, 4))
    plt.plot(frames, donor_I, label='Donor', color='blue')
    plt.plot(frames, acceptor_I, label='Acceptor', color='red')
    plt.title(f"Intensity Plot | Frames: {total_frames}, Intensity: {total_intensity}, Background: {background}, Xd: {Xd}")
    plt.xlabel("Frame")
    plt.ylabel("Photon Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 图 2：FRET效率图
    plt.figure(figsize=(10, 4))
    plt.plot(frames, ideal_E, label='Ideal FRET', color='green')
    plt.plot(frames, noisy_E, label='Noisy FRET', color='orange')

    # 标出 stimulate_frame（紫色虚线）和 changepoint（黑色实线）
    plt.axvline(x=stimulate_frame, color='purple', linestyle='--', label='Stimulate Frame')
    if changepoint_frame > 0:
        plt.axvline(x=changepoint_frame, color='black', linestyle='-', label='Changepoint')

    plt.title(f"FRET Efficiency | Frames: {total_frames}, Intensity: {total_intensity}, Background: {background}, Xd: {Xd}")
    plt.xlabel("Frame")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



generate_multiple_trajectories(500)