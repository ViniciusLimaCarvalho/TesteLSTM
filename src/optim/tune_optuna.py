"""
Otimização de hiperparâmetros do ConvLSTM com Optuna.

Como rodar:
    python -m src.optim.tune_optuna --folder data/raw/V3.A1_CSV --trials 40

Resultados ficam em results/optuna/optuna_convlstm.db (SQLite, retomável).
Para inspecionar via dashboard:
    optuna-dashboard sqlite:///results/optuna/optuna_convlstm.db
"""
import argparse
import gc
import json
import os
import sys
from datetime import datetime

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from src.api.lstm_core import LSTMCSVFolderModel, _ThermalNet


_FRAMES_CACHE: dict = {}


def _load_once(folder: str) -> np.ndarray:
    """Lê os CSVs da pasta uma única vez por processo."""
    folder = os.path.abspath(folder)
    if folder not in _FRAMES_CACHE:
        helper = LSTMCSVFolderModel()
        _FRAMES_CACHE[folder] = helper._load_folder(folder)
    return _FRAMES_CACHE[folder]


def _build_objective(folder: str, val_frac: float, device: torch.device):
    def objective(trial: optuna.Trial) -> float:
        seq_length  = trial.suggest_int("seq_length", 2, 8)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [2, 4, 8])
        epochs      = trial.suggest_int("epochs", 5, 30)

        data = _load_once(folder)
        n_total = data.shape[0]
        n_val = max(seq_length + 2, int(n_total * val_frac))
        n_train = n_total - n_val
        if n_train <= seq_length + 1:
            raise optuna.TrialPruned("frames insuficientes para o split")

        helper = LSTMCSVFolderModel(seq_length=seq_length, device=device)
        helper.height, helper.width = data.shape[1], data.shape[2]
        helper.flat_size = helper.height * helper.width

        flat_train = helper._normalize(data[:n_train])

        denom = max(helper.temp_max - helper.temp_min, 1e-8)
        val_np = data[n_train:].astype(np.float32)
        flat_val = torch.from_numpy(
            2 * ((val_np - helper.temp_min) / denom) - 1
        ).reshape(n_val, -1)

        train_seqs = helper._make_sequences(flat_train)
        val_seqs = helper._make_sequences(flat_val)
        if not train_seqs or not val_seqs:
            raise optuna.TrialPruned("sequências vazias após split")

        loader = DataLoader(train_seqs, batch_size=batch_size, shuffle=False)
        net = _ThermalNet(helper.height, helper.width, hidden_size, kernel_size).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        try:
            for epoch in range(epochs):
                net.train()
                for seq, label in loader:
                    seq = seq.to(device)
                    label = label.reshape(-1, helper.flat_size).to(device)
                    opt.zero_grad()
                    loss_fn(net(seq), label).backward()
                    opt.step()

                net.eval()
                err_sum = 0.0
                with torch.no_grad():
                    for seq, label in val_seqs:
                        pred = net(seq.unsqueeze(0).to(device)).cpu().squeeze(0)
                        err_sum += (pred - label.squeeze(0)).abs().mean().item()
                val_mae_norm = err_sum / len(val_seqs)
                val_mae_real = val_mae_norm * (helper.temp_max - helper.temp_min) / 2

                best_val = min(best_val, val_mae_real)
                trial.report(val_mae_real, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except torch.cuda.OutOfMemoryError as e:
            raise optuna.TrialPruned(f"CUDA OOM: {e}") from None
        finally:
            del net, opt, loader, train_seqs, val_seqs, flat_train, flat_val
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return best_val

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuning de hiperparâmetros ConvLSTM via Optuna")
    parser.add_argument("--folder", default="data/raw/V3.A1_CSV",
                        help="Pasta com CSVs térmicos")
    parser.add_argument("--trials", type=int, default=40, help="Número de trials")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fração final dos frames usada como validação")
    parser.add_argument("--study-name", default="convlstm_thermal_v2",
                        help="Use um nome novo se mudar o espaço de busca categórico")
    parser.add_argument("--storage", default="sqlite:///results/optuna/optuna_convlstm.db",
                        help="URL de storage do Optuna (SQLite por padrão)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu | cuda (auto se omitido)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Pasta não encontrada: {args.folder}")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    storage_path = args.storage.replace("sqlite:///", "")
    if storage_path and not storage_path.startswith(":"):
        os.makedirs(os.path.dirname(storage_path) or ".", exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_warmup_steps=3),
    )

    print(f"Estudo: {args.study_name} | device: {device} | trials: {args.trials}")
    print(f"Storage: {args.storage}")

    study.optimize(
        _build_objective(args.folder, args.val_frac, device),
        n_trials=args.trials,
        show_progress_bar=True,
        catch=(torch.cuda.OutOfMemoryError, RuntimeError),
        gc_after_trial=True,
    )

    print("\n=== Melhor trial ===")
    print(f"val_mae (real): {study.best_value:.4f}")
    print("hyperparams:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_json = os.path.join(os.path.dirname(storage_path) or ".", "best_params.json")
    payload = {
        "study_name": args.study_name,
        "best_value": float(study.best_value),
        "best_trial_number": study.best_trial.number,
        "n_trials_total": len(study.trials),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "params": study.best_trial.params,
    }
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nMelhores hiperparâmetros salvos em: {best_json}")


if __name__ == "__main__":
    main()
