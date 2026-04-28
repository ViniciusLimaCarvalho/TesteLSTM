# LSTM Thermal CSV Folder API

API REST que treina um LSTM sobre uma pasta de CSVs (cada CSV = uma matriz térmica / um frame) e prevê o próximo frame da sequência.

## Endpoints

- `POST /train` — treina o modelo com os CSVs da pasta. Salva os pesos em `results/models/lstm_csv_folder.pt`.
- `POST /predict` — prevê um frame. Salva CSV (`results/predictions/`) e heatmap turbo JPG (`results/figures/`).
- `POST /predict_stacked` — prevê todas as sequências possíveis e empilha os frames previstos verticalmente em um único CSV + grid de heatmaps JPG.

## Persistência

O modelo treinado é gravado em `results/models/lstm_csv_folder.pt` e recarregado automaticamente no startup da API. Como `./results` é volume Docker, o modelo persiste entre `docker-compose down/up`.

## Timestamp

Cada CSV tem o timestamp embutido no nome (`..._YYYYMMDD_HHMMSSmmm_..._.csv`), parseado com precisão de minutos.

No `POST /predict`, o campo opcional `target_timestamp` permite escolher qual frame prever:

```json
{
  "folder": "data/raw/V3.A1_CSV",
  "target_timestamp": "2026-01-08T02:00",
  "save_outputs": true
}
```

Aceita formato ISO (`2026-01-08T02:00`) ou com espaço (`2026-01-08 02:00`).

**Restrições**:
- O timestamp **precisa existir** entre os arquivos da pasta (predição auto-regressiva de timestamps futuros não é suportada nesta versão).
- Precisa haver pelo menos `seq_length` frames anteriores ao timestamp escolhido.
- Sem `target_timestamp`, o último frame da pasta é usado.

Erros de validação retornam HTTP 400 com mensagem descritiva.

## Executando

### Docker

```bash
sudo docker-compose up -d --build
```

API disponível em `http://localhost:8280`. Documentação interativa em `http://localhost:8280/docs`.

### Local

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Estrutura

```
src/api/
  main.py        # FastAPI app — endpoints /train, /predict, /predict_stacked
  lstm_core.py   # LSTMCSVFolderModel + parsing de timestamp
data/
  raw/V3.A1_CSV/ # CSVs de entrada (cada CSV = uma matriz térmica)
results/
  figures/       # JPGs com heatmaps turbo
  predictions/   # CSVs previstos
  models/        # Checkpoint do LSTM treinado (.pt)
```
