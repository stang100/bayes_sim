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
1. **Generate simulation data**:
   ```bash
   python src/data.py
   ```
2. **Train the Mixture Density Network (MDN)**:
   ```bash
   python src/model.py
   ```
3. **Train policies using PPO**:
   ```bash
   python src/policy_training.py
   ```
4. **Evaluate policies and visualize results**:
   ```bash
   python src/main.py
   ```

## Expected Outputs
After running `main.py`, you will see:
- **Total rewards comparison** (Uniform vs. BayesSim policy)
- **Learning curve plots**
- **Posterior distribution visualizations**
- **Reward distributions for trained policies**

## Future Improvements
- Extend to **more complex environments**.
- Implement **additional inference methods**.
- Improve **MDN training for better posterior accuracy**.

## License
This project is open-source and available for modification.


