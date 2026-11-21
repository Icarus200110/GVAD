import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Conv1DAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 10, window_size: int = 16, latent_dim: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.latent_dim = latent_dim
        reduced_len = window_size // 4
        flat_dim = 64 * reduced_len

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.encoder_fc = nn.Linear(flat_dim, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=2, stride=2),
        )
        self._reduced_len = reduced_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.encoder_conv(h)
        h = h.flatten(start_dim=1)
        z = self.encoder_fc(h)
        d = self.decoder_fc(z)
        d = d.view(-1, 64, self._reduced_len)
        out = self.decoder_conv(d)
        return out.transpose(1, 2)


def train_autoencoder(
    windows: np.ndarray,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Conv1DAutoencoder:
    torch_device = torch.device(device)
    model = Conv1DAutoencoder(
        in_channels=windows.shape[2],
        window_size=windows.shape[1],
        latent_dim=16,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(torch_device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


@torch.no_grad()
def score_autoencoder(
    model: Conv1DAutoencoder,
    windows: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    if windows.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    torch_device = torch.device(device)
    model.eval()
    model.to(torch_device)
    scores = []
    tensor_x = torch.from_numpy(windows.astype(np.float32))
    for i in range(0, tensor_x.shape[0], batch_size):
        batch = tensor_x[i : i + batch_size].to(torch_device)
        recon = model(batch)
        mse = torch.mean((batch - recon) ** 2, dim=(1, 2))
        scores.append(mse.cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)
