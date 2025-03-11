import numpy as np
import torch
from stable_baselines3 import PPO
from data import generate_pendulum_data
from model import train_mdn, infer_posterior
from policy_training import CustomPendulumEnv
from visuals import plot_policy_comparison, plot_learning_curve, plot_reward_distribution

# Load trained policies
uniform_policy = PPO.load("ppo_uniform")
bayessim_policy = PPO.load("ppo_bayessim")

# Define test environment 
test_env = CustomPendulumEnv(mass=1.5, length=1.2)

def evaluate_policy(policy, env, steps=500):
    """Runs the policy in the environment and computes total reward."""
    obs, _ = env.reset()
    total_reward = 0
    reward_history = []

    for _ in range(steps):
        action, _ = policy.predict(obs)
        obs, reward, _, _, _ = env.step(action)
        total_reward += reward
        reward_history.append(total_reward)
    
    return total_reward, reward_history

# Evaluate both policies
uniform_reward, uniform_reward_history = evaluate_policy(uniform_policy, test_env)
bayessim_reward, bayessim_reward_history = evaluate_policy(bayessim_policy, test_env)

print(f"Performance Comparison:\n - Uniform Prior Policy: {uniform_reward}\n - BayesSim Posterior Policy: {bayessim_reward}")

# Generate Visualizations
plot_policy_comparison(uniform_reward, bayessim_reward)
plot_learning_curve(uniform_reward_history, "Uniform Prior Policy")
plot_learning_curve(bayessim_reward_history, "BayesSim Posterior Policy")
plot_reward_distribution([uniform_reward_history, bayessim_reward_history], ["Uniform Prior Policy", "BayesSim Posterior Policy"])
