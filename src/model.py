import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class RandomFourierFeatures(nn.Module):
    """Random Fourier Feature Layer to approximate an RBF kernel."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * 2 * np.pi, requires_grad=False)
        self.b = nn.Parameter(torch.rand(output_dim) * 2 * np.pi, requires_grad=False)

    def forward(self, x):
        if x.shape[1] != self.W.shape[0]:  
            raise ValueError(f"Input dimension {x.shape[1]} does not match expected {self.W.shape[0]}")
        return torch.sqrt(torch.tensor(2.0 / self.W.shape[1])) * torch.cos(torch.matmul(x, self.W) + self.b)

class MDN(nn.Module):
    """Mixture Density Network with Random Fourier Features."""
    def __init__(self, input_dim=2, output_dim=2, num_components=5, rff_dim=64):
        super(MDN, self).__init__()
        self.rff = RandomFourierFeatures(input_dim, rff_dim)
        self.hidden = nn.Linear(rff_dim, 32)
        self.activation = nn.ReLU()
        self.pi = nn.Linear(32, num_components)  # Mixing coefficients
        self.mu = nn.Linear(32, num_components * output_dim)  # Means
        self.sigma = nn.Linear(32, num_components * output_dim)  # Standard deviations
        self.num_components = num_components
        self.output_dim = output_dim



    def forward(self, x):
        x = self.rff(x)  # Apply RFF transformation
        h = self.activation(self.hidden(x))
        pi = torch.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.num_components, self.output_dim)
        sigma = torch.exp(self.sigma(h)).view(-1, self.num_components, self.output_dim)
        return pi, mu, sigma

def train_mdn(params, data, epochs=200, batch_size=32):
    """Train the MDN model."""
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(params, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MDN(input_dim=data.shape[1], output_dim=params.shape[1], num_components=5, rff_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            pi, mu, sigma = model(x_batch)
            loss = -torch.mean(torch.log(torch.sum(torch.exp(torch.distributions.Normal(mu, sigma).log_prob(y_batch.unsqueeze(1))) * pi.unsqueeze(-1), dim=1) + 1e-8))
            loss.backward()
            optimizer.step()
    return model


def infer_posterior(model, observation, num_samples=100):
    """Infer posterior by sampling from the full learned distribution."""
    with torch.no_grad():
        observation = torch.tensor(observation, dtype=torch.float32)
        
        pi, mu, sigma = model(observation)
        component = torch.multinomial(pi[0], num_samples, replacement=True)
        mu_selected = mu[0].gather(0, component.unsqueeze(-1).expand(-1, mu.shape[2]))
        sigma_selected = sigma[0].gather(0, component.unsqueeze(-1).expand(-1, sigma.shape[2]))
        return torch.normal(mu_selected, sigma_selected).numpy()
