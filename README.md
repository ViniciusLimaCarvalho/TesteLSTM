# LSTM Thermal CSV Folder API

API REST que treina um LSTM sobre uma pasta de CSVs (cada CSV = uma matriz térmica / um frame) e prevê o próximo frame da sequência.

## Endpoints

- `POST /train` — treina o modelo com os CSVs da pasta. Salva os pesos em `results/models/lstm_csv_folder.pt`.
- `POST /predict` — prevê um frame. Salva CSV (`results/predictions/`) e heatmap turbo JPG (`results/figures/`).
- `POST /predict_stacked` — prevê todas as sequências possíveis e empilha os frames previstos verticalmente em um único CSV + grid de heatmaps JPG. Também salva um gráfico de erro (MAE) por amostra de tempo em `results/figures/erro_csv_folder_stacked.jpg`.

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

**Comportamento**:
- Se o timestamp **existe** entre os arquivos, a predição é comparada com o frame real e o resultado inclui `mae`, `rmse` e `max_pixel_error`.
- Se o timestamp **não existe**, a predição é sintética: o modelo é iterado de forma auto-regressiva a partir do último frame disponível antes do alvo. O número de iterações é estimado pelo intervalo mediano entre frames da pasta. O resultado vem com `synthetic: true` e `autoregressive_steps: N`, sem métricas de erro (não há frame real para comparar).
- Precisa haver pelo menos `seq_length` frames anteriores ao timestamp escolhido (ou ao âncora, no caso sintético).
- Sem `target_timestamp`, o último frame da pasta é usado.

Erros de validação retornam HTTP 400 com mensagem descritiva.

## Erro por amostra (predict_stacked)

`POST /predict_stacked` agora salva também um gráfico `results/figures/erro_csv_folder_stacked.jpg` com o MAE de cada frame previsto vs o real, com timestamps no eixo X. Útil pra ver em quais momentos o modelo erra mais. O JSON da resposta inclui o array `mae_per_frame` com os valores numéricos.

## Temperatura real vs prevista (predict_stacked)

Também é gerado `results/figures/temperatura_csv_folder_stacked.jpg`, com a temperatura média de cada frame ao longo do tempo:

- Pontos verdes — frames reais.
- Linha vermelha — frames previstos. Entre cada par de previsões reais, são inseridas **3 predições sintéticas** geradas auto-regressivamente (a saída do modelo é realimentada como entrada). O resultado é uma curva mais densa e visualmente contínua.

O número de sintéticas entre cada par é configurável via parâmetro `n_synthetic_between` na chamada interna (default 3).

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
