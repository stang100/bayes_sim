# Bayesian Simulation-Based Policy Training

## Overview
This project implements Bayesian Simulation-Based Inference to train reinforcement learning policies for a **custom Pendulum environment**. 

**BayesSim's learned posterior distribution** is also evaluated with respect to traditional **uniform domain randomization**.

## Features
- **Simulation Data Generation**: Generates simulated trajectories for the pendulum with different mass and length values.
- **Bayesian Posterior Inference**: Uses a Mixture Density Network (MDN) to learn a posterior over simulator parameters.
- **Policy Training**:
  - **Uniform Prior Policy**: Trains a PPO policy with randomly sampled simulator parameters.
  - **BayesSim Posterior Policy**: Trains a PPO policy using inferred posterior parameters.
- **Evaluation & Visualization**:
  - Compares performance of **Uniform vs. BayesSim policies**.
  - Plots **learning curves** and **reward distributions**.
  
## Project Structure
```
├── README.md             # Project documentation (this file)
├── requirements.txt      # Dependencies for running the project
├── src                   # Source code folder
│   ├── data.py           # Generates simulation data (mass, length, trajectories)
│   ├── model.py          # Trains MDN to infer posterior parameters
│   ├── policy_training.py # Trains PPO policies using Uniform and BayesSim-based sampling
│   ├── main.py           # Evaluates trained policies and generates visualizations
│   ├── visuals.py        # Functions for plotting policy performance and distributions
```

## Installation
Ensure you have **Python 3.8+** installed. Then install dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Project
1. **Quick Results (Recommended)**:
   This project already contains generated data and trained policies. To see results comparison immediately:
   ```bash
   python src/main.py
   ```

2. **Retrain Policies Only**:
   If you want to retrain the policies with the already generated data in this project directory:
   ```bash
   python src/policy_training.py
   python src/main.py
   ```

3. **Complete Training Pipeline**:
   If you want to run the entire process from scratch:
   ```bash
   python src/data.py              # Generate simulation data
   python src/policy_training.py   # Train policies
   python src/main.py              # Evaluate and visualize results
   ```

## Expected Outputs
After running `main.py`, you will see:
- **Performance comparison** across different pendulum environments (Default, Light Short, Heavy Long)
- **Visualizations of policy comparisons** showing total reward differences
- **Plots** comparing uniform prior policy vs. Flow-BayesSim policy
- **Overall improvement percentage** across all test environments

Note: Results may vary between runs due to the stochastic nature of reinforcement learning training. The Flow-BayesSim approach may sometimes outperform the uniform prior approach and sometimes underperform, depending on training conditions.

## Potential Improvements
- **Increase training stability** by using more timesteps or multiple random seeds
- **Implement ensemble methods** to reduce variance in policy performance
- **Add adaptive parameter sampling** to better explore the parameter space
- **Extend to more complex environments** beyond the pendulum system

## License
This project is open-source and available for modification.


