import numpy as np
import matplotlib.pyplot as plt

class IntensityandEfficiencySimulator:
    def __init__(self, total_frames, stimulate_frame, total_intensity, background, snr, Xd):
        self.total_frame = total_frames
        self.stimulate_frame = stimulate_frame
        self.total_intensity = total_intensity
        self.background = background
        self.snr = snr
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
        state = np.random.choice(self.E1)
        in_state_1 = True
        history = [state] #用来存效率
        state_history = [1] #用来存状态数

        for t in range(1,self.total_frame):
            if in_state_1:
                state_index = np.where(self.E1 == state)[0][0]
                next_state_probability = self.T1[state_index]
                next_state_index = np.random.choice([0,1,2,3],p = next_state_probability)# 按概率选择索引
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
                state_index = np.where(self.E2 == state)[0][0]
                next_state_probability = self.T2[state_index]
                next_state_index = np.random.choice([0,1,2], p = next_state_probability)
                state = self.E2[next_state_index]
                state_history.append(2)
            history.append(state)
        return np.array(history), np.array(state_history)

    def intensity_trajectories(self, efficiency):
        a = []
        d = []
        fret_efficiency = []

        for e in efficiency:
            idealA = e * self.total_intensity + self.background
            idealD = (1 - e) * self.total_intensity + self.background
            idealA += self.Xd * (idealD - self.background)

            scale_factor = self.snr ** 2 / (idealA + idealD)
            scaled_A = idealA * scale_factor
            scaled_D = idealD * scale_factor

            noisy_A = np.random.poisson(scaled_A) / scale_factor
            noisy_D = np.random.poisson(scaled_D) / scale_factor

            current_fret = noisy_A / (noisy_A + noisy_D)
            fret_efficiency.append(current_fret)

            a.append(noisy_A)
            d.append(noisy_D)

        return np.array(a), np.array(d), np.array(fret_efficiency)

    def save_to_txt(self, noisyA, noisyD, reale, history, filename="output.txt"):
        with open(filename, 'w') as f:
            for a, d, e, h in zip(noisyA, noisyD, reale, history):
                f.write(f"{a} {d} {e} {h}\n")
        print(f"Data saved to {filename}")

def generate_random_parameters():
    """Generate random parameters for IntensityandEfficiencySimulator"""
    # total_frames = np.random.randint(300, 1001)
    total_frames = 500  #wxm double check
    stimulate_frame = total_frames // 2
    total_intensity = np.random.randint(80, 151)
    background = 5
    snr = np.random.uniform(3, 7)
    Xd = 0.1
    return (total_frames, stimulate_frame, total_intensity, background, snr, Xd)


def generate_multiple_datasets(N, output_prefix="dataset"):
    ideal_data_list = []
    noisy_data_list = []
    ideal_filename = 'ideal_efficiency.npy'
    noisy_filename = 'noisy_efficiency.npy'

    """Generate N datasets with random parameters"""
    for i in range(N):
        params = generate_random_parameters()
        simulator = IntensityandEfficiencySimulator(*params)

        #ideal_filename = 'test_ideal_efficiency_%s.npy' % params[4]
        #noisy_filename = 'test_noisy_efficiency_%s.npy' % params[4]

        history, state_history = simulator.stimulator()
        noisyA, noisyD, reale = simulator.intensity_trajectories(history)

        # Save to file
        filename = f"{output_prefix}_{i + 1}.txt"
        simulator.save_to_txt(noisyA, noisyD, reale, filename)

        ideal_data_list.append(history)
        noisy_data_list.append(reale)

        print(f"Generated dataset {i + 1} with parameters:")
        print(f"Total frames: {params[0]}, Stimulate frame: {params[1]}, Total intensity: {params[2]}, "
              f"Background: {params[3]}, SNR: {params[4]:.2f}, Xd: {params[5]}")

    ideal_data_array = np.array(ideal_data_list)
    noisy_data_array = np.array(noisy_data_list)
    print(f"shape {ideal_data_array.shape}")
    print(f"shape {noisy_data_array.shape}")

    # Save the data to .npy files
    np.save(ideal_filename, ideal_data_array)
    np.save(noisy_filename, noisy_data_array)

    print(f"Saved {N} datasets to {ideal_filename} and {noisy_filename}")
    print(f"\nSuccessfully generated {N} datasets")

def plot_intensity_state(Ia, Ib, total_frames, state_history, stimulate_frame):
    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.plot(Ia, label="Intensity_A", color="red")
    ax1.set_xlabel("Time (Frames)")
    ax1.set_ylabel("Intensity A", color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(Ib, label="Intensity_B", color="blue")
    ax2.set_ylabel("Intensity B", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    ax3.plot(state_history, label="State", color="black", linestyle="--")
    ax3.set_ylabel("Steady State")
    ax3.set_yticks([1, 2])
    ax3.set_yticklabels(["State 1", "State 2"])
    ax3.tick_params(axis='y', labelcolor='black')

    ax1.axvline(x=stimulate_frame, color='green', linestyle='--', label='Stimulus')
    ax1.set_xlim(0, total_frames)
    ax1.set_ylim(0, max(Ia.max(), Ib.max()) * 1.1)
    ax2.set_ylim(0, max(Ia.max(), Ib.max()) * 1.1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

    plt.title("Intensity and State Over Time")
    plt.tight_layout()
    plt.draw()
    plt.show()


def plot_real_efficiency(real_e, stimulate_frame=None):
    plt.figure(figsize=(10, 5))
    plt.plot(real_e, color='orange', label='Real FRET Efficiency')

    if stimulate_frame is not None:
        plt.axvline(x=stimulate_frame, color='red', linestyle='--', label='Stimulation Start')

    plt.xlabel('Time (Frames)')
    plt.ylabel('FRET Efficiency')
    plt.title('Real FRET Efficiency Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.draw()
    plt.show()


def generate_and_plot_datasets(N):
    """Generate N datasets with plots for each one"""
    for i in range(1, N + 1):
        print(f"\nGenerating dataset {i} of {N}")
        params = generate_random_parameters()
        simulator = IntensityandEfficiencySimulator(*params)
        history, state_history = simulator.stimulator()
        noisyA, noisyD, reale = simulator.intensity_trajectories(history)

        # Save to file
        filename = f"dataset_{i}.txt"
        simulator.save_to_txt(noisyA, noisyD, reale, history, filename)

        # Show parameters
        print(f"Parameters for dataset {i}:")
        print(f"Total frames: {params[0]}, Stimulate frame: {params[1]}, Total intensity: {params[2]}, "
              f"Background: {params[3]}, SNR: {params[4]:.2f}, Xd: {params[5]}")

        # Generate plots
        # plot_intensity_state(noisyA, noisyD, simulator.total_frame, state_history, simulator.stimulate_frame)
        # plot_real_efficiency(reale, simulator.stimulate_frame)

    print(f"\nSuccessfully generated {N} datasets with plots")

def main():
    print("FRET Efficiency Data Generator")
    print("-----------------------------")
    while True:
        try:
            N = int(input("Enter the number of datasets to generate (or 0 to exit): "))
            if N == 0:
                print("Exiting program...")
                break
            if N < 1:
                print("Please enter a positive integer.")
                continue
            #generate_and_plot_datasets(N)
            generate_multiple_datasets(N)
        except ValueError:
            print("Please enter a valid integer.")

if __name__ == "__main__":
    main()



















