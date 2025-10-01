import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

script_start = time.time()

# --- INÍCIO DO SCRIPT DE DIAGNÓSTICO ---

print("--- ETAPA 1: CONFIGURAÇÃO ---")
device = torch.device("cuda")
print(f"Dispositivo selecionado: {device}")

try:
    df = pd.read_csv('input/measures_v2.csv')
    print(f"Arquivo CSV carregado com sucesso.")
    print(f"Formato do DataFrame (linhas, colunas): {df.shape}")
except FileNotFoundError:
    print("ERRO: Arquivo 'input/measures_v2_reduced.csv' não encontrado. Verifique o caminho.")
    exit()

print("\n--- ETAPA 2: PREPARAÇÃO DOS DADOS ---")
target_column = 'stator_winding'
features_columns = [col for col in df.columns if col != target_column]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df[features_columns + [target_column]].values)
data_tensor = torch.FloatTensor(scaled_data)


def create_sequences(input_data, seq_length):
    inout_seq = []
    L = len(input_data)
    for i in range(L - seq_length):
        train_seq = input_data[i:i + seq_length]
        train_label = input_data[i + seq_length:i + seq_length + 1, -1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


seq_length = 10
sequences = create_sequences(data_tensor, seq_length)
print(f"Número total de sequências criadas: {len(sequences)}")

train_size = int(len(sequences) * 0.8)
test_size = len(sequences) - train_size
train_data = sequences[:train_size]
test_data = sequences[train_size:]

# PONTO DE CHECAGEM CRÍTICO 1: Verificar se os conjuntos de dados não estão vazios
print(f"Número de sequências de TREINO: {len(train_data)}")
print(f"Número de sequências de TESTE: {len(test_data)}")

if not test_data:
    print("\nALERTA: O conjunto de dados de teste está VAZIO. Isso pode acontecer se o arquivo CSV for muito pequeno.")
    print("O gráfico final ficará em branco. Aumente o tamanho do CSV ou diminua a porcentagem de treino.")
    exit()

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# --- Definição do Modelo
class LSTMModel(nn.Module):
    def __init__(self, input_size=len(features_columns) + 1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(device)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        return predictions


model = LSTMModel().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n--- ETAPA 3: TREINAMENTO ---")
epochs = 10
model.train()
for i in range(epochs):
    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    if (i + 1) % 5 == 0:
        print(f'Época {i + 1} perda: {single_loss.item()}')
print("Treinamento concluído.")

print("\n--- ETAPA 4: AVALIAÇÃO ---")
model.eval()
predictions = []
actuals = []
batch_count = 0
with torch.no_grad():
    for seq, labels in test_loader:
        batch_count += 1
        seq, labels = seq.to(device), labels.to(device)
        output = model(seq)

        if batch_count == 1:
            # PONTO DE CHECAGEM CRÍTICO 2: Verificar a saída do modelo
            print(f"Formato do tensor de saída do modelo (batch): {output.shape}")
            print(f"Amostra das 5 primeiras previsões (normalizadas) do primeiro lote: \n{output.flatten()[:5]}")
            # Verificar se há valores inválidos (NaN ou Inf)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(
                    "\nALERTA: O modelo está gerando valores inválidos (NaN ou Inf)! Isso pode ser causado por instabilidade no treinamento.")

        predictions.extend(output.cpu().numpy().flatten())
        actuals.extend(labels.cpu().numpy().flatten())

# PONTO DE CHECAGEM CRÍTICO 3: Verificar se as listas de resultados foram preenchidas
print(f"\nNúmero total de previsões coletadas: {len(predictions)}")
print(f"Número total de valores reais coletados: {len(actuals)}")

if not predictions:
    print("\nALERTA: A lista de previsões está VAZIA. O loop de avaliação pode não ter sido executado.")
    exit()

print("\n--- ETAPA 5: PÓS-PROCESSAMENTO E PLOTAGEM ---")
# Reverter a normalização
dummy_array_pred = np.zeros((len(predictions), len(features_columns) + 1))
dummy_array_pred[:, -1] = predictions
y_pred_inv = scaler.inverse_transform(dummy_array_pred)[:, -1]

dummy_array_actual = np.zeros((len(actuals), len(features_columns) + 1))
dummy_array_actual[:, -1] = actuals
y_test_inv = scaler.inverse_transform(dummy_array_actual)[:, -1]

# PONTO DE CHECAGEM CRÍTICO 4: Verificar os valores finais antes de plotar
print(f"Amostra das 5 primeiras previsões (escala original): {y_pred_inv[:5]}")
print(f"Amostra dos 5 primeiros valores reais (escala original): {y_test_inv[:5]}")

plt.figure(figsize=(15, 6))
plt.plot(y_test_inv[:300], label='Valores Reais', color='blue')
plt.plot(y_pred_inv[:300], label='Valores Preditos', color='red', linestyle='--')
plt.title('Predição de Temperatura com PyTorch LSTM (Diagnóstico)')
plt.xlabel('Amostras de Tempo')
plt.ylabel('Temperatura')
plt.legend()
plt.grid(True)
print("\nTentando exibir o gráfico...")
plt.show()
script_end = time.time()
total_script_time = script_end - script_start
print(f"\nTempo total de execução do script: {total_script_time:.4f} segundos")
print("Script finalizado.")