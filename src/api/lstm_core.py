import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import time
import io
import base64
import h5py
from PIL import Image
import torchvision.transforms as transforms


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def _save_base64_image(b64_data: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(base64.b64decode(b64_data))


# ─── Modelo 1: CSV / Tabular ──────────────────────────────────────────────────

class _CSVNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 100, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out[:, -1, :])


class LSTMCSVModel:
    def __init__(
        self,
        seq_length: int = 10,
        hidden_size: int = 100,
        epochs: int = 10,
        lr: float = 0.001,
        batch_size: int = 64,
        device=None,
    ):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.feature_cols = None

    def _make_sequences(self, data):
        seqs = []
        for i in range(len(data) - self.seq_length):
            seqs.append((data[i:i + self.seq_length], data[i + self.seq_length:i + self.seq_length + 1, -1]))
        return seqs

    def train(self, csv_path: str, target_column: str = 'stator_winding') -> dict:
        start = time.time()
        df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in df.columns if c != target_column]
        cols = self.feature_cols + [target_column]

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = self.scaler.fit_transform(df[cols].values)
        data_t = torch.FloatTensor(scaled)

        seqs = self._make_sequences(data_t)
        if not seqs:
            raise ValueError("Dados insuficientes para criar sequências.")

        train_data = seqs[:int(len(seqs) * 0.8)]
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

        input_size = len(cols)
        self.model = _CSVNet(input_size, self.hidden_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        losses = []
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for seq, label in loader:
                seq, label = seq.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(seq), label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        return {
            "epochs": self.epochs,
            "final_loss": round(losses[-1], 6),
            "loss_history": [round(v, 6) for v in losses],
            "training_time_s": round(time.time() - start, 2),
            "device": str(self.device),
        }

    def predict(self, csv_path: str, target_column: str = 'stator_winding', save_path: str = None) -> dict:
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute /csv/train primeiro.")

        df = pd.read_csv(csv_path)
        cols = self.feature_cols + [target_column]
        scaled = self.scaler.transform(df[cols].values)
        data_t = torch.FloatTensor(scaled)

        seqs = self._make_sequences(data_t)
        test_data = seqs[int(len(seqs) * 0.8):]
        loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        preds, actuals = [], []
        self.model.eval()
        with torch.no_grad():
            for seq, label in loader:
                preds.extend(self.model(seq.to(self.device)).cpu().numpy().flatten())
                actuals.extend(label.numpy().flatten())

        n = len(cols)
        dummy_p = np.zeros((len(preds), n)); dummy_p[:, -1] = preds
        dummy_a = np.zeros((len(actuals), n)); dummy_a[:, -1] = actuals
        y_pred = self.scaler.inverse_transform(dummy_p)[:, -1]
        y_true = self.scaler.inverse_transform(dummy_a)[:, -1]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(y_true[:300], label='Valores Reais', color='blue')
        ax.plot(y_pred[:300], label='Valores Preditos', color='red', linestyle='--')
        ax.set_title('Predição de Temperatura - LSTM (CSV)')
        ax.set_xlabel('Amostras de Tempo')
        ax.set_ylabel('Temperatura')
        ax.legend()
        ax.grid(True)
        img_b64 = _fig_to_base64(fig)

        if save_path:
            _save_base64_image(img_b64, save_path)

        return {"mae": round(mae, 4), "rmse": round(rmse, 4), "plot_base64": img_b64}


# ─── Modelo 2: H5 Matrizes Térmicas ──────────────────────────────────────────

class _ThermalNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


class LSTMThermalModel:
    def __init__(
        self,
        seq_length: int = 3,
        hidden_size: int = 128,
        epochs: int = 1,
        lr: float = 0.001,
        batch_size: int = 4,
        img_height: int = 480,
        img_width: int = 640,
        device=None,
    ):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.flat_size = img_height * img_width
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.temp_min = None
        self.temp_max = None
        self.sensor_min = None
        self.sensor_range = None

    def _fill_nan(self, arr: np.ndarray) -> np.ndarray:
        for col in range(arr.shape[1]):
            for row in range(1, arr.shape[0]):
                if np.isnan(arr[row, col]):
                    arr[row, col] = arr[row - 1, col]
        for col in range(arr.shape[1]):
            for row in range(arr.shape[0] - 2, -1, -1):
                if np.isnan(arr[row, col]):
                    arr[row, col] = arr[row + 1, col]
        return np.nan_to_num(arr, nan=0.0)

    def _make_sequences(self, combined, thermal):
        seqs = []
        for i in range(len(combined) - self.seq_length):
            seqs.append((combined[i:i + self.seq_length], thermal[i + self.seq_length:i + self.seq_length + 1]))
        return seqs

    def _load_h5(self, h5_path: str):
        with h5py.File(h5_path, 'r') as hf:
            data_matrix = hf['matrizes_termicas'][:]
            sensor_values = hf['sensores/block4_values'][:]
        return data_matrix, sensor_values

    def _prepare_tensors(self, data_matrix, sensor_values):
        sensor_values = self._fill_nan(sensor_values.copy())
        if self.sensor_min is None:
            self.sensor_min = sensor_values.min(axis=0, keepdims=True)
            sensor_range = sensor_values.max(axis=0, keepdims=True) - self.sensor_min
            sensor_range[sensor_range == 0] = 1.0
            self.sensor_range = sensor_range
        sensor_norm = (sensor_values - self.sensor_min) / self.sensor_range
        sensor_t = torch.FloatTensor(sensor_norm)

        data_t = torch.FloatTensor(data_matrix)
        if self.temp_min is None:
            self.temp_min = data_t.min().item()
            self.temp_max = data_t.max().item()
        data_t = 2 * ((data_t - self.temp_min) / (self.temp_max - self.temp_min)) - 1
        data_t = data_t.view(data_t.size(0), -1)

        combined = torch.cat([data_t, sensor_t], dim=1)
        return combined, data_t

    def train(self, h5_path: str) -> dict:
        start = time.time()
        data_matrix, sensor_values = self._load_h5(h5_path)
        combined, data_t = self._prepare_tensors(data_matrix, sensor_values)

        seqs = self._make_sequences(combined, data_t)
        if not seqs:
            raise ValueError("Dados insuficientes para criar sequências.")
        train_data = seqs[:-1]

        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        input_size = combined.size(1)
        self.model = _ThermalNet(input_size, self.hidden_size, self.flat_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        losses = []
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for seq, label in loader:
                seq = seq.to(self.device)
                label = label.view(-1, self.flat_size).to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(seq), label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        return {
            "epochs": self.epochs,
            "final_loss": round(losses[-1], 6),
            "loss_history": [round(v, 6) for v in losses],
            "training_time_s": round(time.time() - start, 2),
            "device": str(self.device),
        }

    def predict(self, h5_path: str, save_path: str = None) -> dict:
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute /h5/train primeiro.")

        data_matrix, sensor_values = self._load_h5(h5_path)
        combined, data_t = self._prepare_tensors(data_matrix, sensor_values)
        seqs = self._make_sequences(combined, data_t)
        if not seqs:
            raise ValueError("Dados insuficientes para previsão.")

        seq_test, label_real = seqs[-1]
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_test.unsqueeze(0).to(self.device))

        def denorm(t_flat):
            img = t_flat.cpu().view(self.img_height, self.img_width).numpy()
            return ((img + 1) / 2) * (self.temp_max - self.temp_min) + self.temp_min

        mat_pred = denorm(pred)
        mat_real = denorm(label_real)

        error_map = np.abs(mat_real - mat_pred)
        mae = float(np.mean(error_map))
        rmse = float(np.sqrt(np.mean((mat_real - mat_pred) ** 2)))
        max_err = float(np.max(error_map))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].set_title("Real (Ground Truth)")
        im0 = axes[0].imshow(mat_real, cmap='turbo')
        plt.colorbar(im0, ax=axes[0], label='Temperatura')
        axes[0].axis('off')

        axes[1].set_title(f"Previsão (MAE: {mae:.2f})")
        im1 = axes[1].imshow(mat_pred, cmap='turbo')
        plt.colorbar(im1, ax=axes[1], label='Temperatura')
        axes[1].axis('off')

        axes[2].set_title(f"Mapa de Erro (Máx: {max_err:.2f})")
        im2 = axes[2].imshow(error_map, cmap='Reds')
        plt.colorbar(im2, ax=axes[2], label='Erro Absoluto (Graus)')
        axes[2].axis('off')

        img_b64 = _fig_to_base64(fig)
        if save_path:
            _save_base64_image(img_b64, save_path)

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "max_pixel_error": round(max_err, 4),
            "plot_base64": img_b64,
        }


# ─── Modelo 3: Sequências de Imagens RGB ─────────────────────────────────────

class _ImageNet(nn.Module):
    def __init__(self, flat_size: int, hidden_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(flat_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, flat_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


class LSTMImagesModel:
    def __init__(
        self,
        seq_length: int = 3,
        hidden_size: int = 128,
        epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 4,
        img_height: int = 400,
        img_width: int = 400,
        device=None,
    ):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = 3
        self.flat_size = img_height * img_width * 3
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def _load_images(self, folder: str) -> np.ndarray:
        imgs = []
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(os.path.join(folder, fn)).convert('RGB')
                    imgs.append(self._transform(img).view(-1).numpy())
                except Exception:
                    pass
        return np.array(imgs)

    def _make_sequences(self, data):
        seqs = []
        for i in range(len(data) - self.seq_length):
            seqs.append((data[i:i + self.seq_length], data[i + self.seq_length:i + self.seq_length + 1]))
        return seqs

    def train(self, images_folder: str) -> dict:
        start = time.time()
        data = self._load_images(images_folder)
        if len(data) == 0:
            raise ValueError("Nenhuma imagem válida encontrada.")

        data_t = torch.FloatTensor(data)
        seqs = self._make_sequences(data_t)
        if not seqs:
            raise ValueError("Imagens insuficientes para criar sequências.")

        train_data = seqs[:-1]
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        self.model = _ImageNet(self.flat_size, self.hidden_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        losses = []
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for seq, label in loader:
                seq = seq.to(self.device)
                label = label.view(-1, self.flat_size).to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(seq), label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        return {
            "epochs": self.epochs,
            "final_loss": round(losses[-1], 6),
            "loss_history": [round(v, 6) for v in losses],
            "training_time_s": round(time.time() - start, 2),
            "device": str(self.device),
        }

    def predict(self, images_folder: str, save_path: str = None) -> dict:
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute /images/train primeiro.")

        data = self._load_images(images_folder)
        data_t = torch.FloatTensor(data)
        seqs = self._make_sequences(data_t)
        if not seqs:
            raise ValueError("Imagens insuficientes para previsão.")

        seq_test, label_real = seqs[-1]
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_test.unsqueeze(0).to(self.device))

        def proc(t):
            img = t.cpu().view(self.img_channels, self.img_height, self.img_width)
            img = img.permute(1, 2, 0).numpy()
            return np.clip((img + 1) / 2, 0, 1)

        img_pred = proc(pred)
        img_real = proc(label_real)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_title("Real (Ground Truth)")
        axes[0].imshow(img_real)
        axes[0].axis('off')
        axes[1].set_title("Previsão LSTM")
        axes[1].imshow(img_pred)
        axes[1].axis('off')

        img_b64 = _fig_to_base64(fig)
        if save_path:
            _save_base64_image(img_b64, save_path)

        return {"plot_base64": img_b64}
