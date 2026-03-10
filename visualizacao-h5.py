import h5py
import matplotlib.pyplot as plt

# Abrir o ficheiro em modo de leitura
with h5py.File('dataset_axia_completo_2d.h5', 'r') as hf:
    # Acede ao dataset (isto não carrega tudo para a memória RAM)
    matrizes = hf['matrizes_termicas']

    # Verifica a forma (shape) total, por exemplo: (150, 480, 640)
    print("Forma do Dataset:", matrizes.shape)

    # Carrega apenas a primeira matriz térmica para a memória
    imagem_termica_0 = matrizes[0]

    # Pode visualizá-la diretamente
    plt.imshow(imagem_termica_0, cmap='turbo')
    plt.colorbar(label='Temperatura')
    plt.title('Imagem Térmica - Amostra 0')
    plt.savefig('imagem_termica_0.png', dpi=100, bbox_inches='tight')
    print("Imagem salva como 'imagem_termica_0.png'")
