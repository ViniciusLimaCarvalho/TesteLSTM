import matplotlib
# Configura o matplotlib para rodar sem precisar de monitor/janela
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py # Substitui PIL e torchvision
import time

start_time = time.time()

# --- CONFIGURAÇÕES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

ARQUIVO_H5 = 'dataset_axia_completo_2d.h5'

# Dimensões exatas das matrizes salvas no HDF5 (sem RGB, apenas 1 canal térmico)
IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_CHANNELS = 1
FLAT_SIZE = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS # 307.200

SEQ_LENGTH = 3
BATCH_SIZE = 4

# --- ETAPA 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS DO HDF5 ---
print("--- Carregando Matrizes Térmicas do HDF5 ---")

if not os.path.exists(ARQUIVO_H5):
    print(f"Ficheiro '{ARQUIVO_H5}' não encontrado.")
    exit()

# Abre o HDF5 e carrega as matrizes
with h5py.File(ARQUIVO_H5, 'r') as hf:
    # Carrega os dados para a memória como float32
    # Nota: Se o ficheiro for gigante (ex: >10GB), teríamos de usar um PyTorch Dataset para ler aos poucos.
    # Para datasets normais, carregar tudo de uma vez é mais rápido.
    data_matrix = hf['matrizes_termicas'][:]

print(f"Matrizes carregadas: {len(data_matrix)}")

if len(data_matrix) == 0:
    print("Nenhum dado válido encontrado no HDF5.")
    exit()

# Converte para Tensor PyTorch
data_tensor = torch.FloatTensor(data_matrix)

# Normalização Térmica para [-1, 1] (Melhora o treino da LSTM)
temp_min = data_tensor.min()
temp_max = data_tensor.max()
print(f"Temperatura Mínima: {temp_min:.2f} | Máxima: {temp_max:.2f}")

data_tensor = 2 * ((data_tensor - temp_min) / (temp_max - temp_min)) - 1

# Achatar (flatten) cada matriz (De 480x640 para 307200) para entrar na LSTM
data_tensor = data_tensor.view(data_tensor.size(0), -1)

# --- ETAPA 2: CRIAÇÃO DE SEQUÊNCIAS ---
def create_sequences(input_data, seq_length):
    inout_seq = []
    L = len(input_data)
    if L <= seq_length: return []

    for i in range(L - seq_length):
        train_seq = input_data[i:i + seq_length]
        train_label = input_data[i + seq_length:i + seq_length + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

sequences = create_sequences(data_tensor, SEQ_LENGTH)
print(f"Sequências criadas: {len(sequences)}")

if len(sequences) == 0:
    print("Erro: Matrizes insuficientes para criar sequência.")
    exit()

test_size = 1  # Apenas 1 para teste visual
train_size = len(sequences) - test_size

train_data = sequences[:train_size]
test_data = sequences[train_size:]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

# --- ETAPA 3: MODELO LSTM ---
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=FLAT_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        return predictions

print(f"Tamanho do vetor de entrada: {FLAT_SIZE}")
model = ImageLSTM(input_size=FLAT_SIZE, output_size=FLAT_SIZE).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- ETAPA 4: TREINAMENTO ---
print("\n--- Iniciando Treinamento ---")
epochs = 5

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

# --- ETAPA 5: SALVAR RESULTADO ---
print("\n--- Gerando e Salvando Imagem ---")
model.eval()

with torch.no_grad():
    if len(test_data) > 0:
        seq_test, label_real = test_data[0]
        seq_test_tensor = seq_test.unsqueeze(0).to(device)

        prediction = model(seq_test_tensor)

        # Processamento para visualização (Adaptado para 1 canal)
        def process_image_for_plot(tensor_flat):
            # Volta para 2D (480, 640)
            img = tensor_flat.cpu().view(IMG_HEIGHT, IMG_WIDTH).numpy()
            # Desnormaliza de [-1, 1] para [0, 1] apenas para plotagem
            img = (img + 1) / 2
            return np.clip(img, 0, 1)

        img_predicted = process_image_for_plot(prediction)
        img_real = process_image_for_plot(label_real)

        # Plotagem e Salvamento com mapa de cores térmico (inferno)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Real (Ground Truth)")
        plt.imshow(img_real, cmap='inferno')
        plt.colorbar(label='Temp Normalizada')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Previsão LSTM")
        plt.imshow(img_predicted, cmap='inferno')
        plt.colorbar(label='Temp Normalizada')
        plt.axis('off')

        nome_arquivo = 'resultado_previsao_h5.png'
        plt.savefig(nome_arquivo)
        print(f"SUCESSO! A imagem foi salva como '{nome_arquivo}' na pasta do projeto.")

        plt.close()
    else:
        print("Sem dados de teste.")

end_time = time.time()
print(f"Tempo total: {end_time - start_time:.4f} segundos")