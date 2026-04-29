import matplotlib
matplotlib.use('Agg')

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lstm_core import LSTMCSVFolderModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "models", "lstm_csv_folder.pt")

app = FastAPI(
    title="LSTM Thermal CSV Folder API",
    description=(
        "API para treinar e prever matrizes térmicas a partir de uma pasta de CSVs.\n\n"
        "Cada CSV na pasta representa uma matriz térmica (um frame). O LSTM aprende a "
        "sequência temporal e prevê o próximo frame.\n\n"
        "**Endpoints:** `POST /train`, `POST /predict` e `POST /predict_stacked`."
    ),
    version="1.0.0",
)

_model = LSTMCSVFolderModel()
if os.path.exists(MODEL_PATH):
    try:
        _model.load(MODEL_PATH)
        print(f"[startup] Modelo carregado de {MODEL_PATH}")
    except Exception as e:
        print(f"[startup] Falha ao carregar {MODEL_PATH}: {e}")


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


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


@app.post("/train")
def train(req: TrainRequest):
    """Treina o LSTM com os CSVs da pasta informada."""
    global _model
    folder = _resolve(req.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    _model = LSTMCSVFolderModel(
        seq_length=req.seq_length,
        hidden_size=req.hidden_size,
        epochs=req.epochs,
        lr=req.lr,
        batch_size=req.batch_size,
    )
    try:
        result = _model.train(folder)
        _model.save(MODEL_PATH)
        result["model_saved_to"] = MODEL_PATH
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """Gera a próxima matriz prevista. Salva CSV e JPG (heatmap turbo) em results/."""
    if _model.model is None:
        raise HTTPException(status_code=400, detail="Modelo não treinado. Chame POST /train primeiro.")

    folder = _resolve(req.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    csv_path = jpg_path = None
    if req.save_outputs:
        csv_path = os.path.join(PROJECT_ROOT, "results", "predictions", "previsao_csv_folder.csv")
        jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "previsao_csv_folder.jpg")

    try:
        return _model.predict(
            folder,
            target_timestamp=req.target_timestamp,
            save_csv_path=csv_path,
            save_jpg_path=jpg_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_stacked")
def predict_stacked(req: PredictRequest):
    """Gera previsões para todas as sequências e empilha verticalmente em um único CSV.
    Salva o CSV empilhado e um JPG com grid de heatmaps turbo dos frames previstos."""
    if _model.model is None:
        raise HTTPException(status_code=400, detail="Modelo não treinado. Chame POST /train primeiro.")

    folder = _resolve(req.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    csv_path = jpg_path = error_jpg_path = temp_jpg_path = None
    if req.save_outputs:
        csv_path = os.path.join(PROJECT_ROOT, "results", "predictions", "previsao_csv_folder_stacked.csv")
        jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "previsao_csv_folder_stacked.jpg")
        error_jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "erro_csv_folder_stacked.jpg")
        temp_jpg_path = os.path.join(PROJECT_ROOT, "results", "figures", "temperatura_csv_folder_stacked.jpg")

    try:
        return _model.predict_stacked(
            folder,
            save_csv_path=csv_path,
            save_jpg_path=jpg_path,
            save_error_jpg_path=error_jpg_path,
            save_temp_jpg_path=temp_jpg_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
