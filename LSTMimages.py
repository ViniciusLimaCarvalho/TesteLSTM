import matplotlib

# Configura o matplotlib para rodar sem precisar de monitor/janela (evita erros gráficos)
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms
import time

start_time = time.time()

# --- CONFIGURAÇÕES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
pasta = 'input_images_2'

# --- REDUZI PARA 128x128 PARA EVITAR O CRASH DE MEMÓRIA ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3  # RGB
FLAT_SIZE = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

SEQ_LENGTH = 3
BATCH_SIZE = 4

# --- ETAPA 1: CARREGAMENTO E PREPARAÇÃO DAS IMAGENS ---
print("--- Carregando Imagens ---")

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_images_from_folder(folder):
    images = []
    if not os.path.exists(folder):
        print(f"Pasta '{folder}' não encontrada.")
        exit()

    filenames = sorted(os.listdir(folder))
    valid_images = [f for f in filenames if f.endswith((".jpg", ".png", ".jpeg"))]

    for filename in valid_images:
        try:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor.view(-1).numpy())
        except Exception as e:
            print(f"Erro ao ler {filename}: {e}")

    return np.array(images)


data_matrix = load_images_from_folder(pasta)
print(f"Imagens carregadas: {len(data_matrix)}")

if len(data_matrix) == 0:
    print("Nenhuma imagem válida encontrada.")
    exit()

data_tensor = torch.FloatTensor(data_matrix)


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
    print("Erro: Imagens insuficientes para criar sequência.")
    exit()

test_size = 1  # Apenas 1 para teste visual
train_size = len(sequences) - test_size

train_data = sequences[:train_size]
test_data = sequences[train_size:]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)


# Test loader não é estritamente necessário se pegarmos direto do dataset, mas ok manter

# --- ETAPA 3: MODELO LSTM ---
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=FLAT_SIZE):
        super().__init__()
        # Reduzi hidden_layer para 128 para economizar memória
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
epochs = 20  # 20 épocas é suficiente para ver se funciona

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

    if i % 5 == 0:
        print(f'Época {i} Loss: {epoch_loss / len(train_loader):.6f}')

# --- ETAPA 5: SALVAR RESULTADO ---
print("\n--- Gerando e Salvando Imagem ---")
model.eval()

with torch.no_grad():
    if len(test_data) > 0:
        seq_test, label_real = test_data[0]
        seq_test_tensor = seq_test.unsqueeze(0).to(device)

        prediction = model(seq_test_tensor)


        # Processamento para visualização
        def process_image_for_plot(tensor_flat):
            img = tensor_flat.cpu().view(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
            img = img.permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
            img = (img + 1) / 2  # Desnormaliza
            return np.clip(img, 0, 1)


        img_predicted = process_image_for_plot(prediction)
        img_real = process_image_for_plot(label_real)

        # Plotagem e Salvamento
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Real (Ground Truth)")
        plt.imshow(img_real)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Previsão LSTM")
        plt.imshow(img_predicted)
        plt.axis('off')

        nome_arquivo = 'resultado_previsao.png'
        plt.savefig(nome_arquivo)
        print(f"SUCESSO! A imagem foi salva como '{nome_arquivo}' na pasta do projeto.")

        # Limpa a figura da memória
        plt.close()
    else:
        print("Sem dados de teste.")

end_time = time.time()
print(f"Tempo total: {end_time - start_time:.4f} segundos")