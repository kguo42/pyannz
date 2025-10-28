import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def create_model(input_dim, hidden_size, output_dim, depth, activation='tanh'):
    """Create a multi-layer neural network"""
    layers = []

    # Input layer
    layers.append(nn.Linear(input_dim, hidden_size))
    if activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'Sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == 'ReLU':
        layers.append(nn.ReLU())
    elif activation == 'LeakyReLU':
          layers.append(nn.LeakyReLU())
    else:
        raise ValueError(f"Unsupported activation: {activation}")


    # Hidden layers
    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if activation == 'tanh':
          layers.append(nn.Tanh())
        elif activation == 'Sigmoid':
          layers.append(nn.Sigmoid())
        elif activation == 'ReLU':
          layers.append(nn.ReLU())
        elif activation == 'LeakyReLU':
          layers.append(nn.LeakyReLU())


    # Output layer
    layers.append(nn.Linear(hidden_size, output_dim))

    return nn.Sequential(*layers)


class WeightedMSELoss(nn.Module):
    """MSE loss with more weight on larger y values
    Code provided by Claude."""
    def __init__(self, weight_type='linear'):
        super().__init__()
        self.weight_type = weight_type

    def forward(self, pred, target):
        if self.weight_type == 'linear':
            # Weight proportional to |target|
            weights = 1.0 + torch.abs(target)
        elif self.weight_type == 'sqrt':
            # Less aggressive weighting
            weights = 1.0 + torch.sqrt(torch.abs(target))
        elif self.weight_type == 'inverse_freq':
            # Weight by inverse frequency (if you know distribution)
            weights = torch.ones_like(target)
            # Add your custom weighting logic here

        squared_error = (pred - target) ** 2
        weighted_loss = squared_error * weights
        return weighted_loss.mean()


def train_model(model, x_train, y_train, epochs=2000, lr=0.01, reg_type=None, lambda_reg=0.01, x_val = None, y_val =None, patience = 1000, criterion = nn.MSELoss()):
    """Train the neural network"""

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        # Forward pass
        model.train()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)

        losses.append(loss.item())#pure loss

        # Add regularization if specified
        if reg_type == 'l2':
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2) # we include bias terms in it
            loss += lambda_reg * l2_reg
        elif reg_type == 'l1':
            l1_reg = torch.tensor(0.)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += lambda_reg * l1_reg

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #losses.append(loss.item())

        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_predictions = model(x_val)
                val_loss = criterion(val_predictions, y_val)
                val_losses.append(val_loss.item()) #MSE loss, RMS is the sqrt(MSE)


        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        if epoch % 10 == 0:
            pbar.set_postfix({
                'Train': f'{loss.item():.6f}',
                'Val': f'{val_loss.item():.6f}',
                'Best': f'{best_val_loss:.6f}',
                'Patience': f'{patience_counter}/{patience}'
            })

    return losses, val_losses


