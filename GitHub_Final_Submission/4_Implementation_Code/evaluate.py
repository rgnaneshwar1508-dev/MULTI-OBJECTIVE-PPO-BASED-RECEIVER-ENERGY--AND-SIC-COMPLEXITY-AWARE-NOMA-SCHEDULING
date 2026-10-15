"""
evaluate.py
-----------
Compares the trained PPO scheduler against the baseline schedulers.
Updated to reflect the new fairness weight in the combined score calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from noma_env import NOMASchedulingEnv
from baselines import random_scheduler, greedy_channel_scheduler, fixed_cluster_scheduler

N_EPISODES = 30
N_USERS = 6
N_CLUSTERS = 3

def run_episode_with_policy(env, policy_fn, is_ppo=False, model=None):
    obs, _ = env.reset()
    totals = {"throughput": [], "energy_cost": [], "sic_stages": [], "fairness": []}
    done = False
    
    while not done:
        if is_ppo:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = policy_fn(env)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        totals["throughput"].append(info["throughput"])
        totals["energy_cost"].append(info["energy_cost"])
        totals["sic_stages"].append(info["sic_stages"])
        totals["fairness"].append(info["fairness"])

    return {k: np.mean(v) for k, v in totals.items()}

def main():
    rng = np.random.default_rng(42)

    env = NOMASchedulingEnv(
        n_users=N_USERS,
        n_clusters=N_CLUSTERS,
        w_throughput=1.0,
        w_energy=2.5,
        w_sic=1.0,
        w_fairness=0.5,
        max_steps=50,
    )

    model = PPO.load("ppo_noma_model")
    results = {"Random": [], "Greedy": [], "Fixed": [], "PPO (ours)": []}

    for ep in range(N_EPISODES):
        results["Random"].append(run_episode_with_policy(
            env, lambda e: random_scheduler(N_USERS, N_CLUSTERS, rng)))
        results["Greedy"].append(run_episode_with_policy(
            env, lambda e: greedy_channel_scheduler(e.channel_gains, N_CLUSTERS)))
        results["Fixed"].append(run_episode_with_policy(
            env, lambda e: fixed_cluster_scheduler(N_USERS, N_CLUSTERS)))
        results["PPO (ours)"].append(run_episode_with_policy(
            env, None, is_ppo=True, model=model))

    summary = {}
    for method, episode_results in results.items():
        summary[method] = {
            metric: np.mean([ep[metric] for ep in episode_results])
            for metric in ["throughput", "energy_cost", "sic_stages", "fairness"]
        }

    # Updated weights to match the new environment training setup
    W_T, W_E, W_S, W_F = 1.0, 2.5, 1.0, 0.5
    for method, metrics in summary.items():
        metrics["combined_score"] = (
            W_T * metrics["throughput"]
            - W_E * metrics["energy_cost"]
            - W_S * metrics["sic_stages"]
            + W_F * metrics["fairness"]
        )

    print(f"\n=== RESULTS SUMMARY (averaged over {N_EPISODES} episodes) ===\n")
    print(f"{'Method':<15}{'Throughput':>12}{'Energy':>12}{'SIC Stages':>12}{'Fairness':>12}{'Combined':>12}")
    for method, metrics in summary.items():
        print(f"{method:<15}{metrics['throughput']:>12.3f}{metrics['energy_cost']:>12.3f}"
              f"{metrics['sic_stages']:>12.3f}{metrics['fairness']:>12.3f}{metrics['combined_score']:>12.3f}")
    
    print("\n('Combined' = throughput - 2.5*energy - 1.0*sic_stages + 0.5*fairness)")

    methods = list(summary.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metric_titles = {
        "throughput": "Average Throughput (higher = better)",
        "energy_cost": "Average Energy Cost (lower = better)",
        "sic_stages": "Average SIC Stages (lower = better)",
        "fairness": "Jain's Fairness Index (higher = better)",
        "combined_score": "Combined Score (higher = better)",
    }

    for ax, (metric, title) in zip(axes.flat, metric_titles.items()):
        values = [summary[m][metric] for m in methods]
        colors = ["#888888", "#4C72B0", "#DD8452", "#55A868"]
        ax.bar(methods, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)

    axes.flat[-1].axis("off")  
    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150)
    print("\nSaved comparison chart to comparison_results.png")

if __name__ == "__main__":
    main()