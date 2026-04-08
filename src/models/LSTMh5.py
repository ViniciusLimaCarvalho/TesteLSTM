import matplotlib

# Configura o matplotlib para rodar sem precisar de monitor/janela
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import time

start_time = time.time()

# --- CONFIGURAÇÕES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# Pega o caminho absoluto da pasta onde o script atual está (src/models)
diretorio_script = os.path.dirname(os.path.abspath(__file__))

# Volta duas pastas para chegar na raiz do projeto (TesteLSTM)
raiz_projeto = os.path.dirname(os.path.dirname(diretorio_script))

# Monta o caminho exato para o arquivo h5
ARQUIVO_H5 = os.path.join(raiz_projeto, 'data', 'processed', 'dataset_axia_completo_2d.h5')
print(f"Buscando arquivo em: {ARQUIVO_H5}")

# Dimensões exatas das matrizes salvas no HDF5
IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_CHANNELS = 1
FLAT_SIZE = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

SEQ_LENGTH = 3
BATCH_SIZE = 4

# --- ETAPA 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS DO HDF5 ---
print("--- Carregando Matrizes Térmicas e Sensores do HDF5 ---")

if not os.path.exists(ARQUIVO_H5):
    print(f"Ficheiro '{ARQUIVO_H5}' não encontrado.")
    exit()

with h5py.File(ARQUIVO_H5, 'r') as hf:
    data_matrix = hf['matrizes_termicas'][:]

    # Lê os dados dos sensores (block4_values contém as 18 colunas numéricas)
    sensor_values = hf['sensores/block4_values'][:]          # shape: (N, 18)
    sensor_names  = [x.decode() for x in hf['sensores/block4_items'][:]]

print(f"Matrizes carregadas: {len(data_matrix)}")
print(f"Sensores disponíveis ({len(sensor_names)}): {sensor_names}")

if len(data_matrix) == 0:
    print("Nenhum dado válido encontrado no HDF5.")
    exit()

# --- PROCESSAMENTO DOS SENSORES ---
# Forward-fill: propaga o último valor válido para os NaN seguintes
for col in range(sensor_values.shape[1]):
    for row in range(1, sensor_values.shape[0]):
        if np.isnan(sensor_values[row, col]):
            sensor_values[row, col] = sensor_values[row - 1, col]

# Backward-fill: preenche NaN restantes no início com o primeiro valor válido
for col in range(sensor_values.shape[1]):
    for row in range(sensor_values.shape[0] - 2, -1, -1):
        if np.isnan(sensor_values[row, col]):
            sensor_values[row, col] = sensor_values[row + 1, col]

# Se ainda restar NaN (coluna inteiramente vazia), substitui por 0
sensor_values = np.nan_to_num(sensor_values, nan=0.0)

nan_remaining = np.sum(np.isnan(sensor_values))
print(f"NaN restantes nos sensores após preenchimento: {nan_remaining}")

NUM_SENSORS = sensor_values.shape[1]

# Normaliza cada sensor independentemente para [0, 1]
sensor_min = sensor_values.min(axis=0, keepdims=True)   # (1, 18)
sensor_max = sensor_values.max(axis=0, keepdims=True)   # (1, 18)
sensor_range = sensor_max - sensor_min
sensor_range[sensor_range == 0] = 1.0                   # evita divisão por zero
sensor_norm = (sensor_values - sensor_min) / sensor_range

sensor_tensor = torch.FloatTensor(sensor_norm)           # (N, 18)

# --- NORMALIZAÇÃO TÉRMICA ---
data_tensor = torch.FloatTensor(data_matrix)

temp_min_orig = data_tensor.min().item()
temp_max_orig = data_tensor.max().item()
print(f"Temperatura Mínima Global: {temp_min_orig:.2f} | Máxima Global: {temp_max_orig:.2f}")

# Normalização para [-1, 1]
data_tensor = 2 * ((data_tensor - temp_min_orig) / (temp_max_orig - temp_min_orig)) - 1
data_tensor = data_tensor.view(data_tensor.size(0), -1)  # (N, FLAT_SIZE)

# Concatena sensores ao vetor térmico: (N, FLAT_SIZE + NUM_SENSORS)
combined_tensor = torch.cat([data_tensor, sensor_tensor], dim=1)
INPUT_SIZE = FLAT_SIZE + NUM_SENSORS
print(f"Tamanho do vetor de entrada (térmico + sensores): {INPUT_SIZE}")


# --- ETAPA 2: CRIAÇÃO DE SEQUÊNCIAS ---
def create_sequences(input_data, thermal_data, seq_length):
    """
    input_data:   tensor (N, INPUT_SIZE)  – térmico + sensores
    thermal_data: tensor (N, FLAT_SIZE)   – apenas térmico (label)
    """
    inout_seq = []
    L = len(input_data)
    if L <= seq_length:
        return []

    for i in range(L - seq_length):
        train_seq   = input_data[i:i + seq_length]                         # (seq_len, INPUT_SIZE)
        train_label = thermal_data[i + seq_length:i + seq_length + 1]      # (1, FLAT_SIZE)
        inout_seq.append((train_seq, train_label))
    return inout_seq


sequences = create_sequences(combined_tensor, data_tensor, SEQ_LENGTH)
print(f"Sequências criadas: {len(sequences)}")

if len(sequences) == 0:
    print("Erro: Matrizes insuficientes para criar sequência.")
    exit()

test_size  = 1
train_size = len(sequences) - test_size

train_data = sequences[:train_size]
test_data  = sequences[train_size:]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)


# --- ETAPA 3: MODELO LSTM ---
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=FLAT_SIZE):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _          = self.lstm(input_seq)
        last_time_step_out   = lstm_out[:, -1, :]
        predictions          = self.linear(last_time_step_out)
        return predictions


model          = ImageLSTM(input_size=INPUT_SIZE, output_size=FLAT_SIZE).to(device)
loss_function  = nn.MSELoss()
optimizer      = torch.optim.Adam(model.parameters(), lr=0.001)

# --- ETAPA 4: TREINAMENTO ---
print("\n--- Iniciando Treinamento ---")
epochs = 1

model.train()
for i in range(epochs):
    epoch_loss = 0
    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.view(-1, FLAT_SIZE).to(device)

        optimizer.zero_grad()
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
        epoch_loss += single_loss.item()

    if i % 5 == 0 or i == epochs - 1:
        print(f'Época {i} Loss: {epoch_loss / len(train_loader):.6f}')

# --- ETAPA 5: AVALIAÇÃO E MEDIÇÃO DE ERRO ---
print("\n--- Gerando Previsão e Medindo Erros ---")
model.eval()

with torch.no_grad():
    if len(test_data) > 0:
        seq_test, label_real = test_data[0]
        seq_test_tensor = seq_test.unsqueeze(0).to(device)

        prediction = model(seq_test_tensor)

        # Função para desnormalizar e voltar para 2D
        def denormalize_and_reshape(tensor_flat, t_min, t_max):
            img_norm = tensor_flat.cpu().view(IMG_HEIGHT, IMG_WIDTH).numpy()
            img_real = ((img_norm + 1) / 2) * (t_max - t_min) + t_min
            return img_real

        # Matrizes na escala real de temperatura
        matriz_prevista = denormalize_and_reshape(prediction, temp_min_orig, temp_max_orig)
        matriz_real     = denormalize_and_reshape(label_real, temp_min_orig, temp_max_orig)

        # --- CÁLCULO DE MÉTRICAS DE ERRO ---
        mapa_erro = np.abs(matriz_real - matriz_prevista)
        mae       = np.mean(mapa_erro)
        rmse      = np.sqrt(np.mean((matriz_real - matriz_prevista) ** 2))
        erro_max  = np.max(mapa_erro)

        print("-" * 30)
        print("MÉTRICAS DE ERRO (Na escala de temperatura original):")
        print(f"MAE (Erro Médio Absoluto): {mae:.2f} graus")
        print(f"RMSE (Erro Quadrático Médio): {rmse:.2f} graus")
        print(f"Erro Máximo num único pixel: {erro_max:.2f} graus")
        print("-" * 30)

        # Plotagem com 3 painéis
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.title("Real (Ground Truth)")
        plt.imshow(matriz_real, cmap='turbo')
        plt.colorbar(label='Temperatura')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"Previsão (MAE: {mae:.2f})")
        plt.imshow(matriz_prevista, cmap='turbo')
        plt.colorbar(label='Temperatura')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Mapa de Erro (Máx: {erro_max:.2f})")
        plt.imshow(mapa_erro, cmap='Reds')
        plt.colorbar(label='Erro Absoluto (Graus)')
        plt.axis('off')

        os.makedirs('results/figures', exist_ok=True)
        nome_arquivo = 'results/figures/resultado_previsao_h5.png'
        plt.savefig(nome_arquivo)
        print(f"\nSUCESSO! A imagem de análise foi salva como '{nome_arquivo}'.")

        plt.close()
    else:
        print("Sem dados de teste.")

end_time = time.time()
print(f"Tempo total: {end_time - start_time:.4f} segundos")
