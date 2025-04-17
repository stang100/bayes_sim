import numpy as np
import torch
from stable_baselines3 import PPO
from data import generate_pendulum_data
from model import train_mdn, infer_posterior
from policy_training import CustomPendulumEnv
from visuals import plot_policy_comparison, plot_learning_curve, plot_reward_distribution

# Load trained policies
try:
    uniform_policy = PPO.load("ppo_uniform")
    flow_policy = PPO.load("ppo_flow")
    print("Loaded pre-trained policies.")
except FileNotFoundError:
    print("No pre-trained policies found. Please run policy_training.py first.")
    exit(1)

# Define test environments with different parameters
test_envs = [
    ("Default", CustomPendulumEnv(mass=1.5, length=1.2)),
    ("Light Short", CustomPendulumEnv(mass=0.5, length=0.7)),
    ("Heavy Long", CustomPendulumEnv(mass=1.8, length=1.8))
]

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

# Evaluate policies on all test environments
print("\nPolicy Performance Across Different Environments:")
print("------------------------------------------------")
overall_uniform_reward = 0
overall_flow_reward = 0

for env_name, env in test_envs:
    uniform_reward, uniform_reward_history = evaluate_policy(uniform_policy, env)
    flow_reward, flow_reward_history = evaluate_policy(flow_policy, env)
    
    improvement = ((flow_reward - uniform_reward) / abs(uniform_reward)) * 100
    
    print(f"\nEnvironment: {env_name} (mass={env.mass}, length={env.length})")
    print(f" - Uniform Prior Policy: {uniform_reward:.2f}")
    print(f" - Flow-BayesSim Policy: {flow_reward:.2f}")
    print(f" - Improvement: {improvement:.2f}%")
    
    overall_uniform_reward += uniform_reward
    overall_flow_reward += flow_reward
    
    # Generate visualizations for each environment
    plot_policy_comparison(uniform_reward, flow_reward, 
                          label2="Flow-BayesSim", 
                          title=f"Performance in {env_name} Environment")
    plot_reward_distribution([uniform_reward_history, flow_reward_history], 
                           ["Uniform Prior Policy", "Flow-BayesSim Policy"],
                           title=f"Reward Distribution in {env_name} Environment")

# Calculate overall improvement
overall_improvement = ((overall_flow_reward - overall_uniform_reward) / abs(overall_uniform_reward)) * 100

print("\nOverall Performance (Average across all environments):")
print(f" - Uniform Prior Policy: {overall_uniform_reward/len(test_envs):.2f}")
print(f" - Flow-BayesSim Policy: {overall_flow_reward/len(test_envs):.2f}")
print(f" - Overall Improvement: {overall_improvement:.2f}%")
