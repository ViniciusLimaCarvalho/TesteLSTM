import matplotlib
matplotlib.use('Agg')

import os
import io
import time
import base64
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def _parse_timestamp(filename: str) -> Optional[datetime]:
    """Extrai datetime (precisão de minutos) do padrão '..._YYYYMMDD_HHMMSSmmm_...csv'."""
    parts = os.path.basename(filename).split('_')
    if len(parts) < 4:
        return None
    date_str, time_str = parts[2], parts[3]
    if len(date_str) != 8 or len(time_str) < 4:
        return None
    try:
        return datetime(
            year=int(date_str[:4]),
            month=int(date_str[4:6]),
            day=int(date_str[6:8]),
            hour=int(time_str[:2]),
            minute=int(time_str[2:4]),
        )
    except ValueError:
        return None


def _fig_to_base64(fig, fmt: str = 'jpg') -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


class _ThermalNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


class LSTMCSVFolderModel:
    """LSTM treinado sobre uma pasta de CSVs onde cada CSV é uma matriz térmica."""

    def __init__(
        self,
        seq_length: int = 3,
        hidden_size: int = 128,
        epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 4,
        device=None,
    ):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.temp_min = None
        self.temp_max = None
        self.height = None
        self.width = None
        self.flat_size = None
        self.timestamps: list = []

    def _load_folder(self, folder: str) -> np.ndarray:
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.csv'))
        if not files:
            raise ValueError(f"Nenhum CSV encontrado em {folder}.")

        frames = []
        timestamps = []
        for fn in files:
            arr = pd.read_csv(os.path.join(folder, fn), header=None).values.astype(np.float32)
            frames.append(arr)
            timestamps.append(_parse_timestamp(fn))

        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise ValueError(f"Formatos divergentes entre CSVs: {shapes}")

        self.timestamps = timestamps
        return np.stack(frames, axis=0)

    def _normalize(self, data: np.ndarray) -> torch.Tensor:
        if self.temp_min is None:
            self.temp_min = float(data.min())
            self.temp_max = float(data.max())
        denom = max(self.temp_max - self.temp_min, 1e-8)
        norm = 2 * ((data - self.temp_min) / denom) - 1
        return torch.FloatTensor(norm).reshape(data.shape[0], -1)

    def _make_sequences(self, flat: torch.Tensor):
        seqs = []
        for i in range(len(flat) - self.seq_length):
            seqs.append((flat[i:i + self.seq_length], flat[i + self.seq_length:i + self.seq_length + 1]))
        return seqs

    def train(self, folder: str) -> dict:
        start = time.time()
        data = self._load_folder(folder)
        self.height, self.width = data.shape[1], data.shape[2]
        self.flat_size = self.height * self.width

        flat = self._normalize(data)
        seqs = self._make_sequences(flat)
        if not seqs:
            raise ValueError("Frames insuficientes para criar sequências.")

        train_data = seqs[:-1]
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

        self.model = _ThermalNet(self.flat_size, self.hidden_size, self.flat_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        losses = []
        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for seq, label in loader:
                seq = seq.to(self.device)
                label = label.reshape(-1, self.flat_size).to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(seq), label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        return {
            "epochs": self.epochs,
            "frames_loaded": int(data.shape[0]),
            "frame_shape": [self.height, self.width],
            "final_loss": round(losses[-1], 6),
            "loss_history": [round(v, 6) for v in losses],
            "training_time_s": round(time.time() - start, 2),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nada para salvar — modelo não treinado.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "hyperparams": {
                "seq_length": self.seq_length,
                "hidden_size": self.hidden_size,
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
            },
            "norm": {"temp_min": self.temp_min, "temp_max": self.temp_max},
            "shape": {"height": self.height, "width": self.width, "flat_size": self.flat_size},
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        h = ckpt["hyperparams"]
        self.seq_length = h["seq_length"]
        self.hidden_size = h["hidden_size"]
        self.epochs = h["epochs"]
        self.lr = h["lr"]
        self.batch_size = h["batch_size"]
        self.temp_min = ckpt["norm"]["temp_min"]
        self.temp_max = ckpt["norm"]["temp_max"]
        self.height = ckpt["shape"]["height"]
        self.width = ckpt["shape"]["width"]
        self.flat_size = ckpt["shape"]["flat_size"]
        self.model = _ThermalNet(self.flat_size, self.hidden_size, self.flat_size).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(
        self,
        folder: str,
        target_timestamp=None,
        save_csv_path: str = None,
        save_jpg_path: str = None,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute /train primeiro.")

        data = self._load_folder(folder)
        flat = self._normalize(data)

        if target_timestamp is not None:
            if isinstance(target_timestamp, str):
                target_ts = datetime.fromisoformat(target_timestamp.replace(' ', 'T'))
            else:
                target_ts = target_timestamp
            target_ts = target_ts.replace(second=0, microsecond=0)
            try:
                idx = next(i for i, ts in enumerate(self.timestamps) if ts == target_ts)
            except StopIteration:
                raise ValueError(
                    f"Timestamp {target_ts.isoformat()} não encontrado nos arquivos da pasta."
                )
            if idx < self.seq_length:
                raise ValueError(
                    f"Timestamp em índice {idx} requer {self.seq_length} frames anteriores; "
                    f"apenas {idx} disponíveis."
                )
            seq_test = flat[idx - self.seq_length:idx]
            label_real = flat[idx:idx + 1]
        else:
            seqs = self._make_sequences(flat)
            if not seqs:
                raise ValueError("Frames insuficientes para previsão.")
            seq_test, label_real = seqs[-1]
            idx = len(self.timestamps) - 1

        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_test.unsqueeze(0).to(self.device))

        def denorm(t_flat):
            img = t_flat.detach().cpu().reshape(self.height, self.width).numpy()
            return ((img + 1) / 2) * (self.temp_max - self.temp_min) + self.temp_min

        mat_pred = denorm(pred)
        mat_real = denorm(label_real)
        error_map = np.abs(mat_real - mat_pred)
        mae = float(np.mean(error_map))
        rmse = float(np.sqrt(np.mean((mat_real - mat_pred) ** 2)))
        max_err = float(np.max(error_map))

        target_ts = self.timestamps[idx] if 0 <= idx < len(self.timestamps) else None
        ts_str = target_ts.strftime('%Y-%m-%d %H:%M') if target_ts else 'N/A'

        result = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "max_pixel_error": round(max_err, 4),
            "frame_shape": [self.height, self.width],
            "target_timestamp": target_ts.isoformat() if target_ts else None,
        }

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            np.savetxt(save_csv_path, mat_pred, delimiter=',', fmt='%.4f')
            result["csv_saved_to"] = save_csv_path

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat_pred, cmap='turbo')
        ax.set_title(f"Previsão LSTM — {ts_str} (MAE: {mae:.2f})")
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Temperatura')

        if save_jpg_path:
            os.makedirs(os.path.dirname(save_jpg_path), exist_ok=True)
            fig.savefig(save_jpg_path, format='jpg', bbox_inches='tight')
            result["jpg_saved_to"] = save_jpg_path

        result["plot_base64"] = _fig_to_base64(fig, fmt='jpg')
        return result

    def predict_stacked(
        self,
        folder: str,
        save_csv_path: str = None,
        save_jpg_path: str = None,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute /train primeiro.")

        data = self._load_folder(folder)
        flat = self._normalize(data)
        seqs = self._make_sequences(flat)
        if not seqs:
            raise ValueError("Frames insuficientes para previsão.")

        preds, actuals = [], []
        self.model.eval()
        with torch.no_grad():
            for seq, label in seqs:
                pred = self.model(seq.unsqueeze(0).to(self.device))
                preds.append(pred.detach().cpu().reshape(self.height, self.width).numpy())
                actuals.append(label.detach().cpu().reshape(self.height, self.width).numpy())

        scale = self.temp_max - self.temp_min
        preds_d = np.stack([((p + 1) / 2) * scale + self.temp_min for p in preds], axis=0)
        actuals_d = np.stack([((a + 1) / 2) * scale + self.temp_min for a in actuals], axis=0)

        error = np.abs(actuals_d - preds_d)
        mae = float(np.mean(error))
        rmse = float(np.sqrt(np.mean((actuals_d - preds_d) ** 2)))
        max_err = float(np.max(error))

        result = {
            "n_predictions": int(preds_d.shape[0]),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "max_pixel_error": round(max_err, 4),
            "frame_shape": [self.height, self.width],
        }

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            stacked = preds_d.reshape(-1, self.width)
            np.savetxt(save_csv_path, stacked, delimiter=',', fmt='%.4f')
            result["csv_saved_to"] = save_csv_path
            result["csv_layout"] = (
                f"{preds_d.shape[0]} frames empilhados verticalmente "
                f"({preds_d.shape[0]}×{self.height} = {stacked.shape[0]} linhas × {self.width} colunas)"
            )

        pred_timestamps = self.timestamps[self.seq_length:] if self.timestamps else []
        result["target_timestamps"] = [
            ts.isoformat() if ts else None for ts in pred_timestamps
        ]

        n_show = min(16, preds_d.shape[0])
        cols = 4
        rows = (n_show + cols - 1) // cols
        indices = np.linspace(0, preds_d.shape[0] - 1, n_show, dtype=int)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
        axes = np.array(axes).reshape(-1)
        vmin, vmax = float(preds_d.min()), float(preds_d.max())
        for i, idx in enumerate(indices):
            im = axes[i].imshow(preds_d[idx], cmap='turbo', vmin=vmin, vmax=vmax)
            ts = pred_timestamps[idx] if idx < len(pred_timestamps) else None
            title = ts.strftime('%m-%d %H:%M') if ts else f"Frame {int(idx)}"
            axes[i].set_title(title)
            axes[i].axis('off')
        for j in range(n_show, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f"Previsões empilhadas — N={preds_d.shape[0]} | MAE={mae:.2f} | RMSE={rmse:.2f}")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label='Temperatura')

        if save_jpg_path:
            os.makedirs(os.path.dirname(save_jpg_path), exist_ok=True)
            fig.savefig(save_jpg_path, format='jpg', bbox_inches='tight')
            result["jpg_saved_to"] = save_jpg_path

        result["plot_base64"] = _fig_to_base64(fig, fmt='jpg')
        return result
