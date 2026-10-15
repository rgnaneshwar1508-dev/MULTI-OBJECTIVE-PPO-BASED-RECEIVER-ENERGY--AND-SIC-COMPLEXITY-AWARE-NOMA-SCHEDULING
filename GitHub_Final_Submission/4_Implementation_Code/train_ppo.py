"""
train_ppo.py
------------
Trains a PPO agent (via stable-baselines3) on the custom NOMASchedulingEnv.
Expanded timesteps and exploration coefficients to handle the more complex 
environment dynamics (imperfect CSI and interference).
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from noma_env import NOMASchedulingEnv

def make_env():
    return NOMASchedulingEnv(
        n_users=6,
        n_clusters=3,
        w_throughput=1.0,
        w_energy=2.5,       
        w_sic=1.0,          
        w_fairness=0.5,     # Added fairness objective weighting
        max_steps=50,
    )

if __name__ == "__main__":
    env = make_vec_env(make_env, n_envs=4)

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.03,  # Increased for stronger exploration of cluster distributions
    )

    # Scaled up training to allow convergence in the more volatile environment
    TOTAL_TIMESTEPS = 1_000_000  

    print("Starting PPO training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save("ppo_noma_model")
    env.save("vecnormalize_stats.pkl") 
    print("Training complete. Model saved to ppo_noma_model.zip")