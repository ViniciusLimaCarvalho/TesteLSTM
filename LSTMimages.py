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

# Parâmetros da Imagem
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 1
FLAT_SIZE = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

SEQ_LENGTH = 3
BATCH_SIZE = 4

# --- ETAPA 1: CARREGAMENTO E PREPARAÇÃO DAS IMAGENS ---
print("--- Carregando Imagens ---")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        if filename.endswith((".jpg")):
            img = Image.open(os.path.join(folder, filename))
            img_tensor = transform(img)
            images.append(img_tensor.view(-1).numpy())
    return np.array(images)


try:
    if not os.path.exists('input_images'):
        print("Pasta 'input_images' não encontrada.")
    else:
        data_matrix = load_images_from_folder('input_images')
        print(f"Imagens carregadas: {len(data_matrix)}")
except Exception as e:
    print(f"Erro: {e}")
    exit()

data_tensor = torch.FloatTensor(data_matrix)


# --- ETAPA 2: CRIAÇÃO DE SEQUÊNCIAS E DIVISÃO TREINO/TESTE ---

def create_sequences(input_data, seq_length):
    inout_seq = []
    L = len(input_data)
    for i in range(L - seq_length):
        train_seq = input_data[i:i + seq_length]
        train_label = input_data[i + seq_length:i + seq_length + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


sequences = create_sequences(data_tensor, SEQ_LENGTH)
print(f"Número total de sequências criadas: {len(sequences)}")

if len(sequences) == 0:
    print("Erro: Aumente o número de imagens ou diminua o SEQ_LENGTH")
    exit()

test_size = 5
train_size = len(sequences) - test_size

train_data = sequences[:train_size]
test_data = sequences[train_size:]

print(f"Sequências de TREINO: {len(train_data)}")
print(f"Sequências de TESTE: {len(test_data)}")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


# --- ETAPA 3: MODELO (ADAPTADO PARA IMAGEM) ---
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=256, output_size=FLAT_SIZE):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        return predictions


model = ImageLSTM(input_size=FLAT_SIZE, output_size=FLAT_SIZE).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- ETAPA 4: TREINAMENTO ---
print("\n--- Iniciando Treinamento ---")
epochs = 50
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

    if i % 10 == 0:
        print(f'Época {i} Loss: {epoch_loss / len(train_loader):.6f}')

# --- ETAPA 5: AVALIAÇÃO VISUAL ---
print("\n--- Gerando Previsão de Imagem ---")
model.eval()

with torch.no_grad():
    seq_test, label_real = test_data[0]
    seq_test_tensor = seq_test.unsqueeze(0).to(device)
    prediction = model(seq_test_tensor)

    img_predicted = prediction.cpu().view(IMG_HEIGHT, IMG_WIDTH).numpy()
    img_real = label_real.view(IMG_HEIGHT, IMG_WIDTH).numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Imagem Real (Ground Truth)")
    plt.imshow(img_real, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Imagem Prevista pelo LSTM")
    plt.imshow(img_predicted, cmap='gray')
    plt.axis('off')

    plt.show()

end_time = time.time()
print(f"\nTempo total de execução: {end_time - start_time:.4f} segundos")