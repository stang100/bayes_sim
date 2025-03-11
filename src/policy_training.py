import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from data import CustomPendulumEnv, generate_pendulum_data
from model import train_mdn, infer_posterior

# Load or generate data for training MDN
try:
    params = np.load("params.npy")
    data = np.load("data.npy")
except FileNotFoundError:
    params, data = generate_pendulum_data()
    np.save("params.npy", params)
    np.save("data.npy", data)

# Train MDN model before policy training
print("Training Mixture Density Network (MDN)...")
mdn_model = train_mdn(params, data)

# Function to train a policy
def train_policy(param_sampler, policy_name="ppo_policy", timesteps=100000):
    def make_env():
        mass, length = param_sampler()
        return CustomPendulumEnv(mass=mass, length=length)

    env = make_vec_env(make_env, n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(policy_name)

# Uniform Prior Sampler
def sample_uniform():
    return np.random.uniform(0.1, 2.0), np.random.uniform(0.1, 2.0)

# BayesSim Posterior Sampler
def sample_bayessim():
    obs, _ = generate_pendulum_data(n_samples=1)  # Get a test observation
    posterior_samples = infer_posterior(mdn_model, obs, num_samples=1)  # Use trained MDN model
    return posterior_samples[0, 0], posterior_samples[0, 1]

# Train policies
print("Training with uniform prior...")
train_policy(sample_uniform, "ppo_uniform")

print("Training with BayesSim posterior...")
train_policy(sample_bayessim, "ppo_bayessim")
