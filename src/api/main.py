import matplotlib
matplotlib.use('Agg')

import sys
import os
import json
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lstm_core import LSTMCSVFolderModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
OPTUNA_BEST_PARAMS_PATH = os.path.join(PROJECT_ROOT, "results", "optuna", "best_params.json")
_TRAINABLE_KEYS = {"seq_length", "hidden_size", "epochs", "lr", "batch_size", "kernel_size"}
MAX_CACHED_MODELS = 5

app = FastAPI(
    title="ConvLSTM Thermal CSV Folder API",
    description=(
        "API para treinar e prever matrizes térmicas a partir de uma pasta de CSVs.\n\n"
        "Cada CSV na pasta representa uma matriz térmica (um frame). O ConvLSTM aprende a "
        "sequência temporal preservando a estrutura espacial 2D e prevê o próximo frame.\n\n"
        "**Endpoints:** `POST /train`, `POST /predict` e `POST /predict_stacked`."
    ),
    version="1.0.0",
)

_models: OrderedDict = OrderedDict()


def _model_path(folder: str) -> str:
    name = os.path.basename(os.path.normpath(folder))
    return os.path.join(MODELS_DIR, f"{name}.pt")


def _get_model(folder: str) -> LSTMCSVFolderModel:
    name = os.path.basename(os.path.normpath(folder))
    if name in _models:
        _models.move_to_end(name)
        return _models[name]

    path = _model_path(folder)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=400,
            detail=f"Modelo não treinado para '{name}'. Chame POST /train com folder='{folder}' primeiro.",
        )
    m = LSTMCSVFolderModel()
    try:
        m.load(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo '{name}': {e}")

    _models[name] = m
    if len(_models) > MAX_CACHED_MODELS:
        evicted, _ = _models.popitem(last=False)
        print(f"[cache] Modelo '{evicted}' removido do cache (limite {MAX_CACHED_MODELS})")
    print(f"[load] Modelo '{name}' carregado de {path}")
    return _models[name]



class TrainRequest(BaseModel):
    folder: str = Field(
        default="data/raw/V3.A1_CSV",
        description="Pasta com os CSVs (relativa à raiz do projeto ou absoluta).",
    )
    seq_length: int = Field(default=3, ge=1)
    hidden_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=5, ge=1)
    lr: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=4, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    device: str = Field(
        default="cpu",
        description="'cuda' ou 'cpu'. Padrão: 'cpu'.",
    )
    use_optuna_params: bool = Field(
        default=False,
        description=(
            "Se True, sobrescreve seq_length/hidden_size/epochs/lr/batch_size/kernel_size com os "
            "valores em results/optuna/best_params.json (gerados pelo último run do tune_optuna). "
            "folder e device continuam vindo deste request."
        ),
    )


class PredictRequest(BaseModel):
    folder: str = Field(default="data/raw/V3.A1_CSV")
    save_outputs: bool = Field(
        default=True,
        description="Salvar CSV previsto em results/predictions/ e figura JPG em results/figures/.",
    )
    target_timestamp: Optional[str] = Field(
        default=None,
        description=(
            "Timestamp do frame a prever (ex.: '2026-01-08T02:00' ou '2026-01-08 02:00'). "
            "Se existir entre os arquivos, compara com o frame real e devolve MAE/RMSE. "
            "Se não existir, usa predição auto-regressiva a partir do último frame anterior "
            "ao timestamp (sem MAE, prediction sintética). "
            "Padrão: prevê o próximo frame após o último da pasta."
        ),
    )
    save_comparison: bool = Field(
        default=False,
        description=(
            "Salva imagem extra (real | previsto | erro por pixel). "
            "No /predict, retorna 400 se o timestamp solicitado for sintético (sem frame real)."
        ),
    )
    save_synthetics: bool = Field(
        default=False,
        description=(
            "Apenas /predict_stacked: salva cada frame interleaved (reais + 3 sintéticas entre cada par) "
            "como JPG individual numa subpasta de results/figures/ identificada pelo intervalo previsto."
        ),
    )
    csv_output_path: Optional[str] = Field(
        default=None,
        description=(
            "Caminho absoluto onde o CSV previsto será salvo (ex.: '/app/autoinspection/previsao.csv'). "
            "Se não informado, salva em results/predictions/ dentro do container."
        ),
    )


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _load_optuna_best() -> dict:
    if not os.path.exists(OPTUNA_BEST_PARAMS_PATH):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Não há hiperparâmetros do Optuna salvos em {OPTUNA_BEST_PARAMS_PATH}. "
                "Rode primeiro: python -m src.optim.tune_optuna ..."
            ),
        )
    with open(OPTUNA_BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/train")
def train(req: TrainRequest):
    """Treina o ConvLSTM com os CSVs da pasta informada e salva o modelo com o nome do módulo."""
    folder = _resolve(req.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    hp = {
        "seq_length": req.seq_length,
        "hidden_size": req.hidden_size,
        "epochs": req.epochs,
        "lr": req.lr,
        "batch_size": req.batch_size,
        "kernel_size": req.kernel_size,
    }
    optuna_meta = None
    if req.use_optuna_params:
        best = _load_optuna_best()
        for k, v in best.get("params", {}).items():
            if k in _TRAINABLE_KEYS:
                hp[k] = v
        optuna_meta = {
            "study_name": best.get("study_name"),
            "best_value": best.get("best_value"),
            "saved_at": best.get("saved_at"),
        }

    name = os.path.basename(os.path.normpath(folder))
    model = LSTMCSVFolderModel(device=req.device, **hp)
    try:
        result = model.train(folder)
        path = _model_path(folder)
        model.save(path)
        _models[name] = model
        _models.move_to_end(name)
        if len(_models) > MAX_CACHED_MODELS:
            evicted, _ = _models.popitem(last=False)
            print(f"[cache] Modelo '{evicted}' removido do cache (limite {MAX_CACHED_MODELS})")
        result["model_saved_to"] = path
        result["hyperparams_used"] = hp
        if optuna_meta is not None:
            result["loaded_from_optuna"] = optuna_meta
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """Gera a próxima matriz prevista. Salva CSV e JPG (heatmap turbo) em results/."""
    folder = _resolve(req.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    model = _get_model(folder)

    csv_path = jpg_path = comparison_path = None
    if req.save_outputs:
        csv_path = req.csv_output_path or os.path.join(PROJECT_ROOT, "results", "predictions", "previsao.csv")
        jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "previsao_csv_folder.jpg")
    if req.save_comparison:
        comparison_path = os.path.join(
            PROJECT_ROOT, "results", "figures", "comparacao_csv_folder.jpg"
        )

    try:
        return model.predict(
            folder,
            target_timestamp=req.target_timestamp,
            save_csv_path=csv_path,
            save_jpg_path=jpg_path,
            save_comparison_jpg_path=comparison_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_stacked")
def predict_stacked(req: PredictRequest):
    """Gera previsões para todas as sequências e empilha verticalmente em um único CSV.
    Salva o CSV empilhado e um JPG com grid de heatmaps turbo dos frames previstos."""
    folder = _resolve(req.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    model = _get_model(folder)

    csv_path = jpg_path = error_jpg_path = temp_jpg_path = None
    synthetics_folder = comparison_path = None
    if req.save_outputs:
        csv_path = os.path.join(PROJECT_ROOT, "results", "predictions", "previsao_csv_folder_stacked.csv")
        jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "previsao_csv_folder_stacked.jpg")
        error_jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "erro_csv_folder_stacked.jpg")
        temp_jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "temperatura_csv_folder_stacked.jpg")
    if req.save_synthetics:
        synthetics_folder = os.path.join(PROJECT_ROOT, "results", "figures")
    if req.save_comparison:
        comparison_path = os.path.join(
            PROJECT_ROOT, "results", "figures", "comparacao_csv_folder_stacked.jpg"
        )

    try:
        return model.predict_stacked(
            folder,
            save_csv_path=csv_path,
            save_jpg_path=jpg_path,
            save_error_jpg_path=error_jpg_path,
            save_temp_jpg_path=temp_jpg_path,
            save_synthetics_folder=synthetics_folder,
            save_comparison_jpg_path=comparison_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
