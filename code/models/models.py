from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.config import DriftConfig
from utils.helpers import get_device


class EmbeddingAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        h1: int = 512,
        h2: int = 384,
        h3: int = 256,
        h4: int = 128,
        h5: int = 96,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, h3),
            nn.GELU(),
            nn.Linear(h3, h4),
            nn.GELU(),
            nn.Linear(h4, h5),
            nn.GELU(),
            nn.Linear(h5, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h5),
            nn.GELU(),
            nn.Linear(h5, h4),
            nn.GELU(),
            nn.Linear(h4, h3),
            nn.GELU(),
            nn.Linear(h3, h2),
            nn.GELU(),
            nn.Linear(h2, h1),
            nn.GELU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(
    X_train_scaled: np.ndarray,
    input_dim: int,
    config: DriftConfig,
):
    device = get_device()

    model = EmbeddingAutoencoder(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        h1=config.hidden_1,
        h2=config.hidden_2,
        h3=config.hidden_3,
        h4=config.hidden_4,
        h5=config.hidden_5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.train()
    losses = []

    for epoch in range(config.n_epochs):
        epoch_loss = 0.0

        for (xb,) in dataloader:
            xb = xb.to(device)

            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        losses.append(epoch_loss)

        print(f"Epoch {epoch + 1:03d}/{config.n_epochs} | loss={epoch_loss:.6f}")

    return model, losses


def compute_reconstruction_errors(
    model: EmbeddingAutoencoder,
    X_scaled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    device = get_device()

    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        X_hat, latent = model(X_tensor)

        errors = torch.mean((X_hat - X_tensor) ** 2, dim=1).cpu().numpy()
        latent_np = latent.cpu().numpy()

    return errors, latent_np


def compute_reconstruction_errors_and_jsd(
    model: EmbeddingAutoencoder,
    X_scaled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = get_device()
    eps = 1e-12

    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        X_hat, latent = model(X_tensor)

        errors = torch.mean((X_hat - X_tensor) ** 2, dim=1)

        # Convert vectors into valid probability distributions for JSD.
        p = torch.softmax(X_tensor, dim=1)
        q = torch.softmax(X_hat, dim=1)
        m = 0.5 * (p + q)

        js_div = 0.5 * (
            torch.sum(p * torch.log((p + eps) / (m + eps)), dim=1)
            + torch.sum(q * torch.log((q + eps) / (m + eps)), dim=1)
        )

        errors_np = errors.cpu().numpy()
        latent_np = latent.cpu().numpy()
        jsd_np = js_div.cpu().numpy()

    return errors_np, latent_np, jsd_np


def compute_threshold(
    train_errors: np.ndarray,
    std_multiplier: float = 1.5,
) -> float:
    return float(train_errors.mean() + std_multiplier * train_errors.std())
