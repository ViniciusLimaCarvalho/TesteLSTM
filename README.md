# ConvLSTM Thermal CSV Folder API

API REST que treina um ConvLSTM sobre uma pasta de CSVs (cada CSV = uma matriz térmica / um frame) e prevê o próximo frame da sequência. A arquitetura usa convoluções 2D nas portas do LSTM, preservando a estrutura espacial dos frames em vez de achatá-los.

O hiperparâmetro `hidden_size` da request `/train` passa a controlar o número de canais ocultos do ConvLSTM (kernel 3×3 fixo).

## Endpoints

- `POST /train` — treina o modelo com os CSVs da pasta. Salva os pesos em `results/models/{nome_da_pasta}.pt` (um arquivo por módulo).
- `POST /predict` — prevê um frame. Por padrão, salva o CSV em `/app/prediction/` com o nome no mesmo formato dos CSVs de treino (ex.: `1_3_20260108_020000000_VBY_V3.A1_1.csv`) e o heatmap turbo em `results/figures/`. Use `csv_output_path` para sobrescrever o destino.
- `POST /predict_stacked` — prevê todas as sequências possíveis e empilha os frames previstos verticalmente em um único CSV + grid de heatmaps JPG. Também salva um gráfico de erro (MAE) por amostra de tempo em `results/figures/erro_csv_folder_stacked.jpg`.

## Persistência

Cada módulo treinado tem seu próprio checkpoint em `results/models/{nome_da_pasta}.pt` (ex.: `V3.A1_CSV.pt`, `V4.A12_CSV.pt`). Os modelos são carregados sob demanda em um cache LRU em memória (até 5 modelos simultâneos), com fallback automático para o disco. Como `./results` é volume Docker, os pesos persistem entre `docker-compose down/up`.

## Hiperparâmetros via Optuna

A request `/train` aceita `use_optuna_params: true`, que sobrescreve `seq_length`, `hidden_size`, `epochs`, `lr`, `batch_size` e `kernel_size` com os valores em `results/optuna/best_params.json` (gerados pelo último run de `python -m src.optim.tune_optuna`). `folder` e `device` continuam vindo do request.

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
  predictions/   # CSVs previstos (quando csv_output_path não é informado e /app/prediction não existe)
  models/        # Um .pt por módulo treinado
  optuna/        # best_params.json gerado pelo tune_optuna
```
