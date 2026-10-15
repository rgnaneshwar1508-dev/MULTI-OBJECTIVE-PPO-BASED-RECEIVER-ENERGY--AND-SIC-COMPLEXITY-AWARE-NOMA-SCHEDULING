"""
baselines.py
------------
Non-learning baseline schedulers to compare against the trained PPO agent.
Includes random, greedy channel-gain pairing, fixed clustering, and an
optional battery-threshold greedy heuristic.
"""

import numpy as np


def random_scheduler(n_users: int, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly assigns each user to a cluster."""
    return rng.integers(0, n_clusters, size=n_users)


def greedy_channel_scheduler(channel_gains: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Greedy heuristic: sorts users by descending channel gain and distributes
    them round-robin across clusters to balance strong and weak channel users.
    """
    n_users = len(channel_gains)
    sorted_indices = np.argsort(-channel_gains)
    action = np.zeros(n_users, dtype=int)
    for rank, user_idx in enumerate(sorted_indices):
        action[user_idx] = rank % n_clusters
    return action


def fixed_cluster_scheduler(n_users: int, n_clusters: int) -> np.ndarray:
    """Fixed-size baseline: splits users statically into sequential clusters."""
    return np.array([i % n_clusters for i in range(n_users)], dtype=int)


def battery_aware_greedy_scheduler(
    channel_gains: np.ndarray, energy_levels: np.ndarray, n_clusters: int, battery_threshold: float = 0.4
) -> np.ndarray:
    """
    Battery-aware heuristic: places users with battery below the threshold
    into smaller or dedicated clusters to minimize their SIC overhead,
    then distributes remaining users via channel sorting.
    """
    n_users = len(channel_gains)
    action = np.zeros(n_users, dtype=int)

    low_battery_users = np.where(energy_levels < battery_threshold)[0]
    normal_users = np.where(energy_levels >= battery_threshold)[0]

    cluster_counts = np.zeros(n_clusters, dtype=int)

    # Prioritize low-battery users into least-occupied clusters
    for u in low_battery_users:
        target_cluster = int(np.argmin(cluster_counts))
        action[u] = target_cluster
        cluster_counts[target_cluster] += 1

    # Distribute normal users using channel sorting
    if len(normal_users) > 0:
        sorted_normal = normal_users[np.argsort(-channel_gains[normal_users])]
        for u in sorted_normal:
            target_cluster = int(np.argmin(cluster_counts))
            action[u] = target_cluster
            cluster_counts[target_cluster] += 1

    return action