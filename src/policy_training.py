import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from data import CustomPendulumEnv, generate_pendulum_data
from model import train_mdn, infer_posterior

# Load or generate data for training the normalizing flow
try:
    params = np.load("params.npy")
    data = np.load("data.npy")
except FileNotFoundError:
    params, data = generate_pendulum_data()
    np.save("params.npy", params)
    np.save("data.npy", data)

# Train Normalizing Flow model for posterior estimation
print("Training Normalizing Flow model for posterior estimation...")
# flow_model = train_mdn(params, data)
flow_model = train_mdn(params, data, epochs=500, batch_size=64)  # Increased epochs and batch size

# Function to train a policy
def train_policy(param_sampler, policy_name="ppo_policy", timesteps=200000):
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

# Flow-BayesSim Posterior Sampler
def sample_flow_posterior():
    obs, _ = generate_pendulum_data(n_samples=1)  # Get a test observation
    
    # Get multiple samples to better represent the posterior
    posterior_samples = infer_posterior(flow_model, obs, num_samples=10)
    
    # Choose a random sample from the posterior
    idx = np.random.randint(0, posterior_samples.shape[0])
    
    # Ensure the sampled values are within reasonable ranges
    mass = np.clip(posterior_samples[idx, 0], 0.1, 2.0)
    length = np.clip(posterior_samples[idx, 1], 0.1, 2.0)
    
    return mass, length

# Train policies
print("Training with uniform prior...")
train_policy(sample_uniform, "ppo_uniform")

print("Training with Flow-BayesSim posterior...")
train_policy(sample_flow_posterior, "ppo_flow")
