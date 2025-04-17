import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_posterior(posterior_samples, true_params, baseline_samples=None, title="Posterior Distribution"):
    """Plot the learned posterior distributions and compare with baselines."""
    num_params = posterior_samples.shape[2]
    fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 4))
    
    if num_params == 1:
        axes = [axes]
    
    param_names = ["Mass", "Length"]
    
    for i, param_name in enumerate(param_names):
        sns.kdeplot(posterior_samples[:, :, i].flatten(), label="Flow-BayesSim Posterior", color='blue', ax=axes[i])
        
        if baseline_samples is not None:
            sns.kdeplot(baseline_samples[0][:, i], label="Uniform Prior", color='green', ax=axes[i])
            sns.kdeplot(baseline_samples[1][:, i], label="ABC Posterior", color='orange', ax=axes[i])
        
        axes[i].axvline(true_params[i], color='red', linestyle="--", label="True Value")
        axes[i].set_title(f"Posterior Distribution for {param_name}")
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel("Density")
        axes[i].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_true_vs_inferred(true_params, inferred_means, title="True vs Inferred Parameters"):
    """Compare true parameters vs. inferred posterior means."""
    param_names = ["Mass", "Length"]
    
    fig, axes = plt.subplots(1, len(param_names), figsize=(5 * len(param_names), 4))
    
    if len(param_names) == 1:
        axes = [axes]
    
    for i, param_name in enumerate(param_names):
        axes[i].scatter(true_params[:, i], inferred_means[:, i], label="Inferred vs. True", color="blue")
        axes[i].plot([min(true_params[:, i]), max(true_params[:, i])], 
                     [min(true_params[:, i]), max(true_params[:, i])], linestyle="--", color="red", label="Ideal Match")
        axes[i].set_xlabel(f"True {param_name}")
        axes[i].set_ylabel(f"Inferred {param_name}")
        axes[i].set_title(f"True vs. Inferred {param_name}")
        axes[i].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_sampled_parameters(params, title="Parameter Distribution"):
    """Visualize the distribution of sampled simulator parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.histplot(params[:, 0], kde=True, bins=30, ax=axes[0], color="blue")
    axes[0].set_title("Sampled Mass Distribution")
    axes[0].set_xlabel("Mass")
    
    sns.histplot(params[:, 1], kde=True, bins=30, ax=axes[1], color="green")
    axes[1].set_title("Sampled Length Distribution")
    axes[1].set_xlabel("Length")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_policy_comparison(uniform_reward, flow_reward, label1="Uniform Prior Policy", label2="BayesSim Posterior Policy", title="Policy Performance Comparison"):
    """Compare the performance of Uniform and Flow-BayesSim policies."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([label1, label2], [uniform_reward, flow_reward], color=['green', 'blue'])
    ax.set_ylabel("Total Reward (Higher is Better)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.5)
    
    # Calculate improvement percentage
    improvement = ((flow_reward - uniform_reward) / abs(uniform_reward)) * 100
    if improvement > 0:
        improvement_text = f"+{improvement:.2f}%"
    else:
        improvement_text = f"{improvement:.2f}%"
    
    # Add the percentage text above the bars
    height = max(uniform_reward, flow_reward)
    ax.annotate(improvement_text, 
                xy=(1, flow_reward), 
                xytext=(1, flow_reward + (0.05 * abs(height))),
                ha='center', 
                fontweight='bold',
                color='red' if improvement < 0 else 'green')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(rewards, label, title="Policy Learning Curve"):
    """Plot learning curve of policy training over episodes."""
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label=label)
    plt.xlabel("Training Steps")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_reward_distribution(reward_samples, policy_labels, title="Reward Distribution of Policies"):
    """Plot the distribution of rewards achieved by different policies."""
    plt.figure(figsize=(8, 5))
    for rewards, label in zip(reward_samples, policy_labels):
        sns.kdeplot(rewards, label=label)
    plt.xlabel("Total Reward")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()