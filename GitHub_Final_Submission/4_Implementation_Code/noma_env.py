"""
noma_env.py
------------
Updated NOMA environment including Jain's fairness in the reward, imperfect 
Channel State Information (CSI), and multi-cell interference to simulate 
realistic 5G/6G network conditions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class NOMASchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_users: int = 6,
        n_clusters: int = 2,
        bandwidth: float = 1.0,
        noise_power: float = 0.01,
        total_power: float = 1.0,
        energy_per_sic_stage: float = 0.05,
        base_energy: float = 0.01,
        w_throughput: float = 1.0,
        w_energy: float = 2.5,
        w_sic: float = 1.0,
        w_fairness: float = 0.5,      # Added fairness weight
        max_steps: int = 50,
        seed: int | None = None,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_clusters = n_clusters
        self.bandwidth = bandwidth
        self.noise_power = noise_power
        self.total_power = total_power
        self.energy_per_sic_stage = energy_per_sic_stage
        self.base_energy = base_energy

        self.w_throughput = w_throughput
        self.w_energy = w_energy
        self.w_sic = w_sic
        self.w_fairness = w_fairness  

        self.max_steps = max_steps
        self._step_count = 0

        self.action_space = spaces.MultiDiscrete([n_clusters] * n_users)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_users * 2,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.channel_gains = None
        self.energy_levels = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self.channel_gains = self._rng.uniform(0.05, 1.0, size=self.n_users)
        self.energy_levels = self._rng.uniform(0.3, 1.0, size=self.n_users)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action).astype(int)
        self._step_count += 1

        clusters = {c: [] for c in range(self.n_clusters)}
        for user_idx, cluster_id in enumerate(action):
            clusters[cluster_id].append(user_idx)

        total_throughput = 0.0
        total_energy_cost = 0.0
        total_sic_stages = 0
        per_user_rate = np.zeros(self.n_users)

        for cluster_id, user_indices in clusters.items():
            if len(user_indices) == 0:
                continue

            true_gains = self.channel_gains[user_indices]
            
            # Imperfect CSI: Base station allocates power based on estimated gains
            csi_error = self._rng.normal(0, 0.02, size=len(user_indices))
            perceived_gains = np.clip(true_gains + csi_error, 0.01, 1.0)
            
            inv_gains = 1.0 / (perceived_gains + 1e-6)
            power_shares = inv_gains / inv_gains.sum() * self.total_power

            cluster_size = len(user_indices)
            sic_stages_per_user = max(cluster_size - 1, 0)
            total_sic_stages += sic_stages_per_user * cluster_size

            for i, user_idx in enumerate(user_indices):
                gain = true_gains[i] # Physical transmission relies on true gain
                power = power_shares[i]

                other_power = power_shares.sum() - power
                
                # Multi-cell interference from neighboring clusters
                multi_cell_interf = self._rng.uniform(0.0, 0.05)
                
                sinr = (power * gain) / (other_power * gain + self.noise_power + multi_cell_interf)

                rate = self.bandwidth * np.log2(1.0 + sinr)
                per_user_rate[user_idx] = rate
                total_throughput += rate

                energy_cost = self.base_energy + (sic_stages_per_user * self.energy_per_sic_stage)

                urgency = 1.0 - self.energy_levels[user_idx]
                weighted_energy_cost = energy_cost * (1.0 + 2.0 * urgency)
                total_energy_cost += weighted_energy_cost

                self.energy_levels[user_idx] = max(
                    0.0, self.energy_levels[user_idx] - energy_cost * 0.1
                )

        if np.sum(per_user_rate) > 0:
            fairness = (np.sum(per_user_rate) ** 2) / (
                self.n_users * np.sum(per_user_rate ** 2) + 1e-9
            )
        else:
            fairness = 0.0

        # Multi-objective reward now explicitly includes fairness
        reward = (
            self.w_throughput * total_throughput
            - self.w_energy * total_energy_cost
            - self.w_sic * total_sic_stages
            + self.w_fairness * fairness
        )

        self.channel_gains = np.clip(
            self.channel_gains + self._rng.normal(0, 0.05, size=self.n_users),
            0.05,
            1.0,
        )

        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "throughput": total_throughput,
            "energy_cost": total_energy_cost,
            "sic_stages": total_sic_stages,
            "fairness": fairness,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([self.channel_gains, self.energy_levels]).astype(np.float32)