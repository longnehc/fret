import numpy as np
import matplotlib.pyplot as plt

# Global SNR for noise computation
global_SNR = 50  # example default; can be overridden per instance

class EfficiencySimulator:
    def __init__(self, num_frames, stimulus_frame,
                 dark_intensity=5, snr=None):
        self.num_frames = num_frames
        self.stimulus_frame = stimulus_frame
        self.dark_intensity = dark_intensity
        # Use provided SNR or fallback to global
        self.SNR = snr if snr is not None else global_SNR

        # Efficiency states
        self.E1 = np.array([0.2, 0.4, 0.6, 0.8])
        self.E2 = np.array([0.3, 0.5, 0.9])

        # Transition matrices
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

        # Jump probabilities for switching to E2 after stimulus
        self.jump_probabilities = {0.2: 0.1, 0.8: 0.05}

    def simulate(self, with_noise=True):
        """Run the efficiency state simulation and add noise."""
        state = np.random.choice(self.E1)
        in_ss1 = True
        history = [state]
        ss_history = [1]

        for t in range(1, self.num_frames):
            if in_ss1:
                idx = np.where(self.E1 == state)[0][0]
                state = np.random.choice(self.E1, p=self.T1[idx])
                if t >= self.stimulus_frame and state in self.jump_probabilities:
                    if np.random.rand() < self.jump_probabilities[state]:
                        state = np.random.choice(self.E2)
                        in_ss1 = False
                        ss_history.append(2)
                    else:
                        ss_history.append(1)
                else:
                    ss_history.append(1)
            else:
                idx = np.where(self.E2 == state)[0][0]
                state = np.random.choice(self.E2, p=self.T2[idx])
                ss_history.append(2)
            history.append(state)
        if with_noise:
            noisy_history = self.add_noise(np.array(history))
        else:
            noisy_history = np.array(history)
        return noisy_history, np.array(ss_history)

    def add_noise(self, efficiencies):
        """
        For each ideal efficiency value, sample a noisy efficiency x from:
            p(x) ∝ exp(-((x - eff)**2) / (2 * sigma)),
        where sigma = (total_intensity + dark_intensity) / SNR, and
        total_intensity ~ Uniform(500, 1500) per sample.
        Enforce x ∈ [0,1] via rejection sampling (no simple clipping).
        """
        noisy = []
        for eff in efficiencies:
            # Randomize total intensity per frame
            total_intensity = np.random.uniform(500, 1000)
            sigma = (total_intensity + self.dark_intensity) / self.SNR
            std_dev = np.sqrt(sigma)
            # Rejection sampling to enforce 0 <= x <= 1
            while True:
                x = np.random.normal(loc=eff, scale=std_dev)
                if 0.0 <= x <= 1.0:
                    noisy.append(x)
                    break
        return np.array(noisy)


def plot_efficiency_and_state(history, ss_history, num_frames, stimulus_frame):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(history, label='Efficiency', color='blue')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Efficiency')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(ss_history, label='Steady State', color='red', linestyle='--')
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['SS1', 'SS2'])
    ax2.set_ylim(0.5, 2.5)

    ax1.axvline(x=stimulus_frame, color='green', linestyle=':', label='Stimulus')
    plt.title('Efficiency with Truncated Gaussian Sampling')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.show()

# Example usage
def main():
    sim = EfficiencySimulator(num_frames=1000, stimulus_frame=500, dark_intensity=5, snr=1000)
    hist, ss = sim.simulate(with_noise=False)
    plot_efficiency_and_state(hist, ss, 1000, 500)

if __name__ == '__main__':
    main()
