import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class ConditionalCouplingLayer(nn.Module):
    """
    Conditional coupling layer for RealNVP normalizing flow.
    Scales and shifts part of the input based on the other part and conditional input.
    """
    def __init__(self, input_dim, hidden_dim, condition_dim, mask):
        super().__init__()
        self.mask = mask  # Binary mask determining which inputs to transform
        
        # Increased network capacity with better activation
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),  # Extra layer for more capacity
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Tanh to stabilize scale factor
        )
        
        self.shift_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),  # Extra layer for more capacity
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, condition, reverse=False):
        # Apply mask to determine which inputs to transform
        masked_x = x * self.mask
        
        # Concatenate masked input with condition
        concat_input = torch.cat([masked_x, condition], dim=1)
        
        # Compute scale and shift
        s = self.scale_net(concat_input) * (1 - self.mask)  # Scale only applies to masked-out elements
        t = self.shift_net(concat_input) * (1 - self.mask)  # Shift only applies to masked-out elements
        
        if not reverse:
            # Forward transformation: z = x⊙mask + (1-mask)⊙(x⊙exp(s) + t)
            z = masked_x + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = torch.sum(s * (1 - self.mask), dim=1)
        else:
            # Inverse transformation: x = z⊙mask + (1-mask)⊙((z - t) ⊙ exp(-s))
            x = masked_x + (1 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -torch.sum(s * (1 - self.mask), dim=1)
            
        return z if not reverse else x, log_det

class ConditionalRealNVP(nn.Module):
    """
    Conditional RealNVP normalizing flow model for posterior estimation.
    Uses alternating binary masks for coupling layers.
    """
    def __init__(self, input_dim, condition_dim, hidden_dim=128, num_layers=8):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Improved feature extractor for conditioning
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Create better mixing masks for coupling layers
        self.masks = []
        for i in range(num_layers):
            if i % 2 == 0:
                # Checkerboard pattern for better mixing
                mask = torch.ones(input_dim)
                mask[::2] = 0.0  # Set even indices to 0
            else:
                mask = torch.zeros(input_dim)
                mask[::2] = 1.0  # Set even indices to 1
            self.masks.append(mask)
                
        # Create coupling layers
        self.coupling_layers = nn.ModuleList([
            ConditionalCouplingLayer(input_dim, hidden_dim, hidden_dim, mask) 
            for mask in self.masks
        ])
        
    def forward(self, x, condition, reverse=False):
        """
        Forward pass through the flow.
        x: input tensor (batch_size, input_dim)
        condition: conditioning tensor (batch_size, condition_dim)
        reverse: if True, perform inverse transformation
        """
        # Encode condition
        encoded_condition = self.condition_encoder(condition)
        
        log_det_sum = 0
        
        # Process through coupling layers
        layers = reversed(self.coupling_layers) if reverse else self.coupling_layers
        
        for layer in layers:
            x, log_det = layer(x, encoded_condition, reverse=reverse)
            log_det_sum += log_det
            
        return x, log_det_sum
    
    def sample(self, condition, num_samples=1, temperature=1.0):
        """
        Generate samples from the posterior.
        condition: conditioning tensor (batch_size, condition_dim)
        num_samples: number of samples to generate per condition
        temperature: controls randomness in sampling (lower = more deterministic)
        """
        batch_size = condition.shape[0]
        
        # Repeat conditions for multiple samples
        if num_samples > 1:
            # Repeat each condition num_samples times
            condition = condition.repeat_interleave(num_samples, dim=0)
        
        # Sample from base distribution (standard normal)
        z = torch.randn(batch_size * num_samples, self.input_dim, device=condition.device) * temperature
        
        # Transform through inverse flow
        x, _ = self.forward(z, condition, reverse=True)
        
        # Reshape if multiple samples per condition
        if num_samples > 1:
            x = x.view(batch_size, num_samples, self.input_dim)
            
        return x

def train_mdn(params, data, epochs=600, batch_size=64):
    """
    Train the normalizing flow model (replacement for MDN).
    Uses the same interface as the original train_mdn function.
    """
    # Convert data to tensors
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(params, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize flow model with improved hyperparameters
    input_dim = params.shape[1]  # Dimension of parameters (theta)
    condition_dim = data.shape[1]  # Dimension of observations (x)
    model = ConditionalRealNVP(input_dim=input_dim, condition_dim=condition_dim, 
                             hidden_dim=128, num_layers=8)
    
    # Use Adam with improved learning rate schedule
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25, factor=0.5, verbose=True)
    
    print(f"Training normalizing flow model with {epochs} epochs...")
    print(f"Model parameters: input_dim={input_dim}, condition_dim={condition_dim}")
    print(f"Network: hidden_dim=128, num_layers=8")
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50  # Early stopping patience
    
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass through the flow
            z, log_det = model(y_batch, x_batch)
            
            # Compute log-likelihood of z under standard normal
            log_likelihood = -0.5 * torch.sum(z**2, dim=1) - 0.5 * input_dim * np.log(2 * np.pi)
            
            # Full log-likelihood = log_likelihood(z) + log_det
            loss = -(log_likelihood + log_det).mean()
            
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            
        avg_loss = total_loss / len(dataset)
        scheduler.step(avg_loss)
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Final loss: {best_loss:.4f}")
    return model

def infer_posterior(model, observation, num_samples=100):
    """
    Infer posterior by sampling from the conditional normalizing flow.
    Uses the same interface as the original infer_posterior function.
    """
    with torch.no_grad():
        # Convert observation to tensor
        observation = torch.tensor(observation, dtype=torch.float32)
        
        # Generate samples from the posterior
        # Vary temperature to get diverse samples
        temperatures = [0.8, 1.0, 1.2]
        all_samples = []
        
        for temp in temperatures:
            samples = model.sample(observation, num_samples=num_samples//len(temperatures), temperature=temp)
            
            # Handle different dimensions
            if samples.ndim == 3:  # Multiple samples per condition
                all_samples.append(samples[0].numpy())  # Return samples for first observation
            else:
                all_samples.append(samples.numpy())
        
        # Combine samples from different temperatures
        combined_samples = np.vstack(all_samples)
        
        return combined_samples
