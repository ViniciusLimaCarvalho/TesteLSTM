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


class _ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class _ThermalNet(nn.Module):
    """ConvLSTM mantendo o contrato (batch, seq, H*W) -> (batch, H*W)."""

    def __init__(self, height: int, width: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.height = height
        self.width = width
        self.hidden_channels = hidden_channels
        self.cell = _ConvLSTMCell(1, hidden_channels, kernel_size)
        self.out_conv = nn.Conv2d(
            hidden_channels, 1, kernel_size, padding=kernel_size // 2
        )

    def forward(self, x):
        b, s, _ = x.shape
        x = x.reshape(b, s, 1, self.height, self.width)
        h = torch.zeros(b, self.hidden_channels, self.height, self.width, device=x.device)
        c = torch.zeros_like(h)
        for t in range(s):
            h, c = self.cell(x[:, t], h, c)
        return self.out_conv(h).reshape(b, self.height * self.width)


class LSTMCSVFolderModel:
    """LSTM treinado sobre uma pasta de CSVs onde cada CSV é uma matriz térmica."""

    def __init__(
        self,
        seq_length: int = 3,
        hidden_size: int = 128,
        epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 4,
        kernel_size: int = 3,
        device=None,
    ):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.device = device or torch.device("cpu")
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

        self.model = _ThermalNet(
            self.height, self.width, self.hidden_size, self.kernel_size
        ).to(self.device)
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
                "kernel_size": self.kernel_size,
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
        self.kernel_size = h.get("kernel_size", 3)
        self.temp_min = ckpt["norm"]["temp_min"]
        self.temp_max = ckpt["norm"]["temp_max"]
        self.height = ckpt["shape"]["height"]
        self.width = ckpt["shape"]["width"]
        self.flat_size = ckpt["shape"]["flat_size"]
        self.model = _ThermalNet(
            self.height, self.width, self.hidden_size, self.kernel_size
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(
        self,
        folder: str,
        target_timestamp=None,
        save_csv_path: str = None,
        save_jpg_path: str = None,
        save_comparison_jpg_path: str = None,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute /train primeiro.")

        data = self._load_folder(folder)
        flat = self._normalize(data)

        target_ts = None
        is_synthetic = False
        n_steps = 1
        seq_test = None
        label_real = None

        if target_timestamp is not None:
            if isinstance(target_timestamp, str):
                target_ts = datetime.fromisoformat(target_timestamp.replace(' ', 'T'))
            else:
                target_ts = target_timestamp
            target_ts = target_ts.replace(second=0, microsecond=0)

            exact_idx = next(
                (i for i, ts in enumerate(self.timestamps) if ts == target_ts), None
            )
            if exact_idx is not None:
                if exact_idx < self.seq_length:
                    raise ValueError(
                        f"Timestamp em índice {exact_idx} requer {self.seq_length} frames anteriores; "
                        f"apenas {exact_idx} disponíveis."
                    )
                seq_test = flat[exact_idx - self.seq_length:exact_idx]
                label_real = flat[exact_idx:exact_idx + 1]
            else:
                valid = sorted(
                    [(i, ts) for i, ts in enumerate(self.timestamps) if ts is not None],
                    key=lambda p: p[1],
                )
                before = [p for p in valid if p[1] < target_ts]
                if not before:
                    raise ValueError(
                        f"Timestamp {target_ts.isoformat()} é anterior a todos os frames "
                        f"disponíveis na pasta."
                    )
                anchor_idx, anchor_ts = before[-1]
                if anchor_idx + 1 < self.seq_length:
                    raise ValueError(
                        f"Frame âncora em índice {anchor_idx} requer {self.seq_length} frames; "
                        f"apenas {anchor_idx + 1} disponíveis."
                    )
                sorted_ts = [t for _, t in valid]
                deltas = [
                    (sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
                    for i in range(len(sorted_ts) - 1)
                ]
                if not deltas:
                    raise ValueError("Não há intervalos suficientes para inferir o passo temporal.")
                median_step = sorted(deltas)[len(deltas) // 2]
                target_delta = (target_ts - anchor_ts).total_seconds()
                n_steps = max(1, round(target_delta / median_step))
                seq_test = flat[anchor_idx - self.seq_length + 1:anchor_idx + 1]
                is_synthetic = True
        else:
            seqs = self._make_sequences(flat)
            if not seqs:
                raise ValueError("Frames insuficientes para previsão.")
            seq_test, label_real = seqs[-1]
            target_ts = self.timestamps[-1] if self.timestamps else None

        if save_comparison_jpg_path and is_synthetic:
            raise ValueError(
                "Comparação real vs previsto indisponível: timestamp solicitado é "
                "sintético (não há frame real correspondente para comparar)."
            )

        self.model.eval()
        with torch.no_grad():
            if is_synthetic and n_steps > 1:
                seq_window = seq_test.clone()
                for _ in range(n_steps):
                    pred = self.model(seq_window.unsqueeze(0).to(self.device))
                    pred_cpu = pred.detach().cpu().squeeze(0)
                    seq_window = torch.cat([seq_window[1:], pred_cpu.unsqueeze(0)], dim=0)
            else:
                pred = self.model(seq_test.unsqueeze(0).to(self.device))

        def denorm(t_flat):
            img = t_flat.detach().cpu().reshape(self.height, self.width).numpy()
            return ((img + 1) / 2) * (self.temp_max - self.temp_min) + self.temp_min

        mat_pred = denorm(pred)
        ts_str = target_ts.strftime('%Y-%m-%d %H:%M') if target_ts else 'N/A'

        result = {
            "synthetic": is_synthetic,
            "autoregressive_steps": n_steps,
            "frame_shape": [self.height, self.width],
            "target_timestamp": target_ts.isoformat() if target_ts else None,
        }

        if label_real is not None:
            mat_real = denorm(label_real)
            error_map = np.abs(mat_real - mat_pred)
            mae = float(np.mean(error_map))
            rmse = float(np.sqrt(np.mean((mat_real - mat_pred) ** 2)))
            max_err = float(np.max(error_map))
            result["mae"] = round(mae, 4)
            result["rmse"] = round(rmse, 4)
            result["max_pixel_error"] = round(max_err, 4)
            title_metric = f" (MAE: {mae:.2f})"
        else:
            title_metric = f" (sintético, {n_steps} passo{'s' if n_steps > 1 else ''})"

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            np.savetxt(save_csv_path, mat_pred, delimiter=',', fmt='%.4f')
            result["csv_saved_to"] = save_csv_path

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat_pred, cmap='turbo')
        ax.set_title(f"Previsão ConvLSTM — {ts_str}{title_metric}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Temperatura')

        if save_jpg_path:
            os.makedirs(os.path.dirname(save_jpg_path), exist_ok=True)
            fig.savefig(save_jpg_path, format='jpg', bbox_inches='tight')
            result["jpg_saved_to"] = save_jpg_path

        result["plot_base64"] = _fig_to_base64(fig, fmt='jpg')

        if save_comparison_jpg_path and label_real is not None:
            mat_real_cmp = denorm(label_real)
            err_cmp = np.abs(mat_real_cmp - mat_pred)
            vmin = float(min(mat_real_cmp.min(), mat_pred.min()))
            vmax = float(max(mat_real_cmp.max(), mat_pred.max()))
            fig_cmp, axes = plt.subplots(1, 3, figsize=(18, 6))
            im0 = axes[0].imshow(mat_real_cmp, cmap='turbo', vmin=vmin, vmax=vmax)
            axes[0].set_title('Real')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], label='Temperatura', shrink=0.85)
            im1 = axes[1].imshow(mat_pred, cmap='turbo', vmin=vmin, vmax=vmax)
            axes[1].set_title('Previsto')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], label='Temperatura', shrink=0.85)
            im2 = axes[2].imshow(err_cmp, cmap='inferno', vmin=0)
            axes[2].set_title(f'Erro |real - prev|  (max={err_cmp.max():.2f})')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], label='Erro absoluto', shrink=0.85)
            fig_cmp.suptitle(f"Comparação ConvLSTM — {ts_str} | MAE={result['mae']:.2f} | RMSE={result['rmse']:.2f}")
            os.makedirs(os.path.dirname(save_comparison_jpg_path), exist_ok=True)
            fig_cmp.savefig(save_comparison_jpg_path, format='jpg', bbox_inches='tight')
            plt.close(fig_cmp)
            result["comparison_jpg_saved_to"] = save_comparison_jpg_path

        return result

    def predict_stacked(
        self,
        folder: str,
        save_csv_path: str = None,
        save_jpg_path: str = None,
        save_error_jpg_path: str = None,
        save_temp_jpg_path: str = None,
        save_synthetics_folder: str = None,
        save_comparison_jpg_path: str = None,
        n_synthetic_between: int = 3,
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
        mae_per_frame = error.mean(axis=(1, 2))

        result = {
            "n_predictions": int(preds_d.shape[0]),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "max_pixel_error": round(max_err, 4),
            "frame_shape": [self.height, self.width],
            "mae_per_frame": [round(float(v), 4) for v in mae_per_frame],
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

        fig_err, ax_err = plt.subplots(figsize=(12, 5))
        x = np.arange(len(mae_per_frame))
        ax_err.plot(x, mae_per_frame, marker='o', linewidth=1.5, color='#c0392b')
        ax_err.fill_between(x, mae_per_frame, alpha=0.15, color='#c0392b')
        if pred_timestamps and any(ts is not None for ts in pred_timestamps):
            step = max(1, len(x) // 12)
            ax_err.set_xticks(x[::step])
            ax_err.set_xticklabels(
                [pred_timestamps[i].strftime('%m-%d %H:%M') if pred_timestamps[i] else str(i)
                 for i in range(0, len(pred_timestamps), step)],
                rotation=45, ha='right',
            )
            ax_err.set_xlabel('Timestamp')
        else:
            ax_err.set_xlabel('Índice da amostra')
        ax_err.set_ylabel('MAE (real vs previsto)')
        ax_err.set_title(f'Erro médio absoluto por amostra — N={len(mae_per_frame)} | MAE global={mae:.2f}')
        ax_err.grid(True, alpha=0.3)
        fig_err.tight_layout()

        if save_error_jpg_path:
            os.makedirs(os.path.dirname(save_error_jpg_path), exist_ok=True)
            fig_err.savefig(save_error_jpg_path, format='jpg', bbox_inches='tight')
            result["error_jpg_saved_to"] = save_error_jpg_path

        plt.close(fig_err)

        interleaved_d = []
        is_real_mask = []
        with torch.no_grad():
            for i, (seq_in, _) in enumerate(seqs):
                interleaved_d.append(preds_d[i])
                is_real_mask.append(True)
                if i < len(seqs) - 1:
                    seq_window = seq_in.clone()
                    cur_pred_flat = torch.from_numpy(preds[i].reshape(-1))
                    for _ in range(n_synthetic_between):
                        seq_window = torch.cat(
                            [seq_window[1:], cur_pred_flat.unsqueeze(0)], dim=0
                        )
                        new_pred = self.model(seq_window.unsqueeze(0).to(self.device))
                        cur_pred_flat = new_pred.detach().cpu().squeeze(0)
                        syn_norm = cur_pred_flat.reshape(self.height, self.width).numpy()
                        interleaved_d.append(((syn_norm + 1) / 2) * scale + self.temp_min)
                        is_real_mask.append(False)

        interleaved_d = np.stack(interleaved_d, axis=0)
        real_temps = actuals_d.mean(axis=(1, 2))
        pred_temps = interleaved_d.mean(axis=(1, 2))

        real_x = np.arange(len(seqs), dtype=float)
        pred_x = np.empty(len(is_real_mask), dtype=float)
        real_idx = 0
        syn_count = 0
        for k, is_real in enumerate(is_real_mask):
            if is_real:
                pred_x[k] = float(real_idx)
                real_idx += 1
                syn_count = 0
            else:
                syn_count += 1
                pred_x[k] = real_idx - 1 + syn_count / (n_synthetic_between + 1)

        fig_temp, ax_temp = plt.subplots(figsize=(12, 5))
        ax_temp.plot(
            pred_x, pred_temps, '-', linewidth=1.2, color='#c0392b',
            label=f'Previsto (real + {n_synthetic_between} sintéticas entre cada par)',
            alpha=0.85,
        )
        ax_temp.plot(
            real_x, real_temps, 'o-', color='#27ae60',
            label='Real', markersize=6, linewidth=1.2, zorder=5,
        )
        if pred_timestamps and any(ts is not None for ts in pred_timestamps):
            step = max(1, len(real_x) // 12)
            tick_idx = np.arange(0, len(pred_timestamps), step)
            ax_temp.set_xticks(real_x[::step])
            ax_temp.set_xticklabels(
                [pred_timestamps[i].strftime('%m-%d %H:%M') if pred_timestamps[i] else str(i)
                 for i in tick_idx],
                rotation=45, ha='right',
            )
            ax_temp.set_xlabel('Timestamp')
        else:
            ax_temp.set_xlabel('Índice da amostra')
        ax_temp.set_ylabel('Temperatura média do frame')
        ax_temp.set_title(
            f'Temperatura real vs prevista — N={len(seqs)} reais '
            f'+ {n_synthetic_between} sintéticas entre pares'
        )
        ax_temp.legend(loc='best')
        ax_temp.grid(True, alpha=0.3)
        fig_temp.tight_layout()

        if save_temp_jpg_path:
            os.makedirs(os.path.dirname(save_temp_jpg_path), exist_ok=True)
            fig_temp.savefig(save_temp_jpg_path, format='jpg', bbox_inches='tight')
            result["temp_jpg_saved_to"] = save_temp_jpg_path

        plt.close(fig_temp)
        result["n_synthetic_between"] = n_synthetic_between

        if save_synthetics_folder:
            interleaved_ts = []
            real_ts_idx = 0
            syn_idx = 0
            for k, is_real in enumerate(is_real_mask):
                if is_real:
                    interleaved_ts.append(pred_timestamps[real_ts_idx]
                                           if real_ts_idx < len(pred_timestamps) else None)
                    real_ts_idx += 1
                    syn_idx = 0
                else:
                    syn_idx += 1
                    prev_ts = pred_timestamps[real_ts_idx - 1] if real_ts_idx - 1 < len(pred_timestamps) else None
                    next_ts = pred_timestamps[real_ts_idx] if real_ts_idx < len(pred_timestamps) else None
                    if prev_ts and next_ts:
                        delta = (next_ts - prev_ts) / (n_synthetic_between + 1)
                        interleaved_ts.append(prev_ts + delta * syn_idx)
                    else:
                        interleaved_ts.append(None)

            def _ts_tag(ts):
                return ts.strftime('%Y%m%d-%H%M') if ts else 'NA'

            first_ts = next((t for t in interleaved_ts if t is not None), None)
            last_ts = next((t for t in reversed(interleaved_ts) if t is not None), None)
            interval_tag = f"{_ts_tag(first_ts)}_to_{_ts_tag(last_ts)}" if first_ts else f"N{len(interleaved_d)}"
            out_dir = os.path.join(save_synthetics_folder, f"stacked_{interval_tag}")
            os.makedirs(out_dir, exist_ok=True)

            vmin_all, vmax_all = float(interleaved_d.min()), float(interleaved_d.max())
            for k, (frame, ts, is_real) in enumerate(zip(interleaved_d, interleaved_ts, is_real_mask)):
                kind = 'real' if is_real else 'syn'
                ts_label = ts.strftime('%Y-%m-%d %H:%M') if ts else f'idx={k}'
                fig_one, ax_one = plt.subplots(figsize=(8, 6))
                im_one = ax_one.imshow(frame, cmap='turbo', vmin=vmin_all, vmax=vmax_all)
                ax_one.set_title(f"[{kind.upper()}] {ts_label}")
                ax_one.axis('off')
                plt.colorbar(im_one, ax=ax_one, label='Temperatura')
                fname = f"frame_{k:04d}_{kind}_{_ts_tag(ts)}.jpg"
                fig_one.savefig(os.path.join(out_dir, fname), format='jpg', bbox_inches='tight')
                plt.close(fig_one)

            result["synthetics_saved_to"] = out_dir
            result["synthetics_count"] = int(len(interleaved_d))

        if save_comparison_jpg_path:
            n_cmp = min(8, preds_d.shape[0])
            cmp_idx = np.linspace(0, preds_d.shape[0] - 1, n_cmp, dtype=int)
            fig_cmp, axes = plt.subplots(n_cmp, 3, figsize=(15, 4 * n_cmp))
            if n_cmp == 1:
                axes = axes.reshape(1, 3)
            for row, idx in enumerate(cmp_idx):
                real_f = actuals_d[idx]
                pred_f = preds_d[idx]
                err_f = np.abs(real_f - pred_f)
                vmin_pair = float(min(real_f.min(), pred_f.min()))
                vmax_pair = float(max(real_f.max(), pred_f.max()))
                ts = pred_timestamps[idx] if idx < len(pred_timestamps) and pred_timestamps[idx] else f"idx={idx}"
                ts_label = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else ts

                im_r = axes[row, 0].imshow(real_f, cmap='turbo', vmin=vmin_pair, vmax=vmax_pair)
                axes[row, 0].set_title(f"Real — {ts_label}")
                axes[row, 0].axis('off')
                plt.colorbar(im_r, ax=axes[row, 0], shrink=0.85)

                im_p = axes[row, 1].imshow(pred_f, cmap='turbo', vmin=vmin_pair, vmax=vmax_pair)
                axes[row, 1].set_title(f"Previsto — MAE={err_f.mean():.2f}")
                axes[row, 1].axis('off')
                plt.colorbar(im_p, ax=axes[row, 1], shrink=0.85)

                im_e = axes[row, 2].imshow(err_f, cmap='inferno', vmin=0)
                axes[row, 2].set_title(f"Erro |R-P| — max={err_f.max():.2f}")
                axes[row, 2].axis('off')
                plt.colorbar(im_e, ax=axes[row, 2], shrink=0.85)

            fig_cmp.suptitle(
                f"Comparação real vs previsto ({n_cmp} amostras de {preds_d.shape[0]}) — "
                f"MAE global={mae:.2f} | RMSE global={rmse:.2f}"
            )
            fig_cmp.tight_layout()
            os.makedirs(os.path.dirname(save_comparison_jpg_path), exist_ok=True)
            fig_cmp.savefig(save_comparison_jpg_path, format='jpg', bbox_inches='tight')
            plt.close(fig_cmp)
            result["comparison_jpg_saved_to"] = save_comparison_jpg_path

        return result
