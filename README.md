# LSTM Temperature Prediction API

API REST para predição de temperatura em motor elétrico usando modelos LSTM.

## Modelos disponíveis

- **CSV** (`/csv`) — LSTM com dados tabulares de sensores
- **H5** (`/h5`) — LSTM com matrizes térmicas 480×640 + sensores
- **Images** (`/images`) — LSTM com sequências de imagens RGB

## Executando

### Docker

```bash
docker-compose up --build
```

A API fica disponível em `http://localhost:8080`.

### Local

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Documentação interativa

Acesse `/docs` (Swagger UI) para testar os endpoints.

## Estrutura

```
src/api/
  main.py        # FastAPI app com endpoints
  lstm_core.py   # Implementação dos 3 modelos LSTM
data/
  raw/input/     # CSV de sensores
  raw/V3.A1/     # Imagens térmicas RGB
  processed/     # Dataset HDF5 unificado
```
