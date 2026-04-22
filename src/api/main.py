import matplotlib
matplotlib.use('Agg')

import sys
import os

# Garante que src/api está no path para import direto de lstm_core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from lstm_core import LSTMCSVModel, LSTMThermalModel, LSTMImagesModel

# Raiz do projeto (TesteLSTM/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="LSTM Temperature Prediction API",
    description=(
        "API local para predição de temperatura em motor elétrico usando modelos LSTM.\n\n"
        "**Modelos disponíveis:**\n"
        "- `/csv` — LSTM com dados tabulares de sensores (CSV)\n"
        "- `/h5` — LSTM com matrizes térmicas 480×640 + sensores (HDF5)\n"
        "- `/images` — LSTM com sequências de imagens RGB\n\n"
        "**Fluxo:** chame `/train` para treinar, depois `/predict` para gerar previsões."
    ),
    version="1.0.0",
)

# Instâncias dos modelos mantidas em memória durante a sessão do servidor
_models: dict = {
    "csv": LSTMCSVModel(),
    "thermal": LSTMThermalModel(),
    "images": LSTMImagesModel(),
}


# ─── Schemas ──────────────────────────────────────────────────────────────────

class CSVTrainRequest(BaseModel):
    csv_path: str = Field(
        default="data/raw/input/measures_v2.csv",
        description="Caminho para o CSV (relativo à raiz do projeto ou absoluto)",
    )
    target_column: str = Field(default="stator_winding", description="Coluna alvo")
    seq_length: int = Field(default=10, ge=1, description="Comprimento da sequência temporal")
    hidden_size: int = Field(default=100, ge=1, description="Tamanho da camada oculta LSTM")
    epochs: int = Field(default=10, ge=1, description="Número de épocas de treino")
    lr: float = Field(default=0.001, gt=0, description="Taxa de aprendizado")
    batch_size: int = Field(default=64, ge=1, description="Tamanho do batch")


class CSVPredictRequest(BaseModel):
    csv_path: str = Field(default="data/raw/input/measures_v2.csv")
    target_column: str = Field(default="stator_winding")
    save_plot: bool = Field(default=True, description="Salvar gráfico em results/figures/")


class H5TrainRequest(BaseModel):
    h5_path: str = Field(
        default="data/processed/dataset_axia_completo_2d.h5",
        description="Caminho para o arquivo HDF5",
    )
    seq_length: int = Field(default=3, ge=1)
    hidden_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=1, ge=1)
    lr: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=4, ge=1)


class H5PredictRequest(BaseModel):
    h5_path: str = Field(default="data/processed/dataset_axia_completo_2d.h5")
    save_plot: bool = Field(default=True)


class ImagesTrainRequest(BaseModel):
    images_folder: str = Field(
        default="data/raw/V3.A1",
        description="Pasta com as imagens (.jpg/.png)",
    )
    seq_length: int = Field(default=3, ge=1)
    hidden_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=5, ge=1)
    lr: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=4, ge=1)
    img_height: int = Field(default=400, ge=1, description="Altura para redimensionar imagens")
    img_width: int = Field(default=400, ge=1, description="Largura para redimensionar imagens")


class ImagesPredictRequest(BaseModel):
    images_folder: str = Field(default="data/raw/V3.A1")
    save_plot: bool = Field(default=True)


# ─── Utilitário ───────────────────────────────────────────────────────────────

def _resolve(path: str) -> str:
    """Resolve caminhos relativos à raiz do projeto."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


# ─── Rotas gerais ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "LSTM Temperature Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "csv": {"train": "POST /csv/train", "predict": "POST /csv/predict"},
            "thermal_h5": {"train": "POST /h5/train", "predict": "POST /h5/predict"},
            "images": {"train": "POST /images/train", "predict": "POST /images/predict"},
        },
    }


@app.get("/status", tags=["Info"])
def status():
    """Retorna quais modelos já foram treinados nesta sessão."""
    return {
        "csv_trained": _models["csv"].model is not None,
        "thermal_trained": _models["thermal"].model is not None,
        "images_trained": _models["images"].model is not None,
    }


# ─── CSV ──────────────────────────────────────────────────────────────────────

@app.post("/csv/train", tags=["CSV - Sensores Tabulares"])
def csv_train(req: CSVTrainRequest):
    """Treina o modelo LSTM com dados tabulares de sensores (arquivo CSV)."""
    path = _resolve(req.csv_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"CSV não encontrado: {path}")

    _models["csv"] = LSTMCSVModel(
        seq_length=req.seq_length,
        hidden_size=req.hidden_size,
        epochs=req.epochs,
        lr=req.lr,
        batch_size=req.batch_size,
    )
    try:
        result = _models["csv"].train(path, target_column=req.target_column)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/csv/predict", tags=["CSV - Sensores Tabulares"])
def csv_predict(req: CSVPredictRequest):
    """Gera previsões com o modelo CSV treinado. Retorna métricas e gráfico em base64."""
    if _models["csv"].model is None:
        raise HTTPException(status_code=400, detail="Modelo CSV não treinado. Chame POST /csv/train primeiro.")

    path = _resolve(req.csv_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"CSV não encontrado: {path}")

    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "resultado_previsao_csv.png") if req.save_plot else None
    try:
        result = _models["csv"].predict(path, target_column=req.target_column, save_path=save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if save_path:
        result["saved_to"] = save_path
    return result


# ─── H5 Matrizes Térmicas ─────────────────────────────────────────────────────

@app.post("/h5/train", tags=["H5 - Matrizes Térmicas"])
def h5_train(req: H5TrainRequest):
    """Treina o modelo LSTM com matrizes térmicas 480×640 + 18 sensores (arquivo HDF5)."""
    path = _resolve(req.h5_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Arquivo H5 não encontrado: {path}")

    _models["thermal"] = LSTMThermalModel(
        seq_length=req.seq_length,
        hidden_size=req.hidden_size,
        epochs=req.epochs,
        lr=req.lr,
        batch_size=req.batch_size,
    )
    try:
        result = _models["thermal"].train(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/h5/predict", tags=["H5 - Matrizes Térmicas"])
def h5_predict(req: H5PredictRequest):
    """Gera previsão da próxima matriz térmica. Retorna MAE, RMSE e gráfico comparativo em base64."""
    if _models["thermal"].model is None:
        raise HTTPException(status_code=400, detail="Modelo H5 não treinado. Chame POST /h5/train primeiro.")

    path = _resolve(req.h5_path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Arquivo H5 não encontrado: {path}")

    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "resultado_previsao_h5.png") if req.save_plot else None
    try:
        result = _models["thermal"].predict(path, save_path=save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if save_path:
        result["saved_to"] = save_path
    return result


# ─── Imagens RGB ──────────────────────────────────────────────────────────────

@app.post("/images/train", tags=["Images - Sequências RGB"])
def images_train(req: ImagesTrainRequest):
    """Treina o modelo LSTM com sequências de imagens RGB de uma pasta."""
    folder = _resolve(req.images_folder)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    _models["images"] = LSTMImagesModel(
        seq_length=req.seq_length,
        hidden_size=req.hidden_size,
        epochs=req.epochs,
        lr=req.lr,
        batch_size=req.batch_size,
        img_height=req.img_height,
        img_width=req.img_width,
    )
    try:
        result = _models["images"].train(folder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/images/predict", tags=["Images - Sequências RGB"])
def images_predict(req: ImagesPredictRequest):
    """Gera previsão do próximo frame da sequência de imagens. Retorna gráfico comparativo em base64."""
    if _models["images"].model is None:
        raise HTTPException(status_code=400, detail="Modelo de imagens não treinado. Chame POST /images/train primeiro.")

    folder = _resolve(req.images_folder)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {folder}")

    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "resultado_previsao_images.png") if req.save_plot else None
    try:
        result = _models["images"].predict(folder, save_path=save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if save_path:
        result["saved_to"] = save_path
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
