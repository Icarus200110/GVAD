from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.synthesis import perturb_windows
from src.models.autoencoder import Conv1DAutoencoder, score_autoencoder
from src.models.hotelling import HotellingT2


def window_to_40d(windows: np.ndarray) -> np.ndarray:
    mean = np.mean(windows, axis=1)
    std = np.std(windows, axis=1)
    vmin = np.min(windows, axis=1)
    vmax = np.max(windows, axis=1)
    return np.concatenate([mean, std, vmin, vmax], axis=1).astype(np.float32)


class GatingMLP(nn.Module):
    def __init__(self, in_dim: int = 40) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateFusion:
    def __init__(
        self,
        mlp: GatingMLP,
        t2_mean: float = 20.0,
        t2_std: float = 17.3,
        ae_mean: float = 0.294,
        ae_std: float = 0.127,
    ) -> None:
        self.mlp = mlp
        self.t2_mean = float(t2_mean)
        self.t2_std = float(max(t2_std, 1e-8))
        self.ae_mean = float(ae_mean)
        self.ae_std = float(max(ae_std, 1e-8))

    def z_normalize(
        self,
        t2_scores: np.ndarray,
        ae_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        zt2 = (t2_scores - self.t2_mean) / self.t2_std
        zae = (ae_scores - self.ae_mean) / self.ae_std
        return zt2.astype(np.float32), zae.astype(np.float32)

    @torch.no_grad()
    def fuse(
        self,
        windows: np.ndarray,
        t2_scores: np.ndarray,
        ae_scores: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if windows.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        torch_device = torch.device(device)
        self.mlp.eval()
        self.mlp.to(torch_device)
        summ = torch.from_numpy(window_to_40d(windows)).to(torch_device)
        weights = self.mlp(summ).cpu().numpy()
        zt2, zae = self.z_normalize(t2_scores, ae_scores)
        fused = weights[:, 0] * zt2 + weights[:, 1] * zae
        return fused.astype(np.float32), weights.astype(np.float32)

    def state_dict(self) -> Dict:
        return {
            "mlp": self.mlp.state_dict(),
            "t2_mean": self.t2_mean,
            "t2_std": self.t2_std,
            "ae_mean": self.ae_mean,
            "ae_std": self.ae_std,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "GateFusion":
        mlp = GatingMLP(in_dim=40)
        mlp.load_state_dict(state["mlp"])
        return cls(
            mlp=mlp,
            t2_mean=float(state["t2_mean"]),
            t2_std=float(state["t2_std"]),
            ae_mean=float(state["ae_mean"]),
            ae_std=float(state["ae_std"]),
        )


def train_gate(
    t2_model: HotellingT2,
    ae_model: Conv1DAutoencoder,
    windows: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    margin: float = 0.1,
    anom_frac: float = 0.20,
    seed: int = 42,
    device: str = "cpu",
) -> GateFusion:
    t2_base = t2_model.score(windows)
    ae_base = score_autoencoder(ae_model, windows, device=device)

    t2_mean = float(np.mean(t2_base))
    t2_std = float(max(np.std(t2_base), 1e-8))
    ae_mean = float(np.mean(ae_base))
    ae_std = float(max(np.std(ae_base), 1e-8))

    torch_device = torch.device(device)
    mlp = GatingMLP(in_dim=40).to(torch_device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    mlp.train()
    for _ in range(epochs):
        norm_w, anom_w = perturb_windows(windows, rng=rng, frac=anom_frac)
        n_summ = window_to_40d(norm_w)
        a_summ = window_to_40d(anom_w)

        n_zt2 = ((t2_model.score(norm_w) - t2_mean) / t2_std).astype(np.float32)
        n_zae = ((score_autoencoder(ae_model, norm_w, device=device) - ae_mean) / ae_std).astype(np.float32)
        a_zt2 = ((t2_model.score(anom_w) - t2_mean) / t2_std).astype(np.float32)
        a_zae = ((score_autoencoder(ae_model, anom_w, device=device) - ae_mean) / ae_std).astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(n_summ),
            torch.from_numpy(n_zt2),
            torch.from_numpy(n_zae),
            torch.from_numpy(a_summ),
            torch.from_numpy(a_zt2),
            torch.from_numpy(a_zae),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for bn_s, bn_t2, bn_ae, ba_s, ba_t2, ba_ae in loader:
            bn_s = bn_s.to(torch_device)
            bn_t2 = bn_t2.to(torch_device)
            bn_ae = bn_ae.to(torch_device)
            ba_s = ba_s.to(torch_device)
            ba_t2 = ba_t2.to(torch_device)
            ba_ae = ba_ae.to(torch_device)

            wn = mlp(bn_s)
            wa = mlp(ba_s)
            sn = wn[:, 0] * bn_t2 + wn[:, 1] * bn_ae
            sa = wa[:, 0] * ba_t2 + wa[:, 1] * ba_ae

            loss = torch.clamp(margin + sn - sa, min=0.0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    mlp.eval()
    return GateFusion(
        mlp=mlp,
        t2_mean=t2_mean,
        t2_std=t2_std,
        ae_mean=ae_mean,
        ae_std=ae_std,
    )
