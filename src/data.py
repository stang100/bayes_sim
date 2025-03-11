import numpy as np
import gym
from gym.envs.classic_control.pendulum import PendulumEnv

class CustomPendulumEnv(gym.envs.classic_control.PendulumEnv):
    """Custom Pendulum Environment with dynamic mass and length."""
    def __init__(self, mass=1.0, length=1.0):
        super().__init__()
        self.mass = mass
        self.length = length
        self.screen = None

    def reset(self, mass=None, length=None, seed=None, options=None):
        if mass is not None:
            self.mass = mass
        if length is not None:
            self.length = length
        return super().reset(seed=seed, options=options)

def generate_pendulum_data(n_samples=1000):
    """Generate simulated data for training the MDN using (mass, length) as inputs."""
    params = []
    data = []
    
    for _ in range(n_samples):
        mass = np.random.uniform(0.1, 2.0)
        length = np.random.uniform(0.1, 2.0)

        params.append([mass, length])
        data.append([mass, length])

    return np.array(params, dtype=np.float32), np.array(data, dtype=np.float32)


if __name__ == "__main__":
    params, data = generate_pendulum_data()
    np.save("params.npy", params)
    np.save("data.npy", data)
