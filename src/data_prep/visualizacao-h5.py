import h5py
import matplotlib.pyplot as plt
import os

# Abrir o ficheiro em modo de leitura no novo diretório
with h5py.File('data/processed/dataset_axia_completo_2d.h5', 'r') as hf:
    matrizes = hf['matrizes_termicas']
    print("Formato do Dataset:", matrizes.shape)

    imagem_termica_0 = matrizes[0]

    plt.imshow(imagem_termica_0, cmap='turbo')
    plt.colorbar(label='Temperatura')
    plt.title('Imagem Térmica - Amostra 0')

    # Salva na pasta correta de resultados
    os.makedirs('results/figures', exist_ok=True)
    caminho_salvamento = 'results/figures/imagem_termica_0.png'

    plt.savefig(caminho_salvamento, dpi=100, bbox_inches='tight')
    print(f"Imagem salva como '{caminho_salvamento}'")