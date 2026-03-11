import pandas as pd
import glob
import os
import numpy as np
import h5py

# --- CONFIGURAÇÕES ---
ARQUIVO_EXCEL = "data/raw/AD-TF2Y.xlsx"
PASTA_MATRIZES = "data/raw/V3.A1_CSV"

# Garante que a pasta processed exista antes de criar o arquivo
os.makedirs('data/processed', exist_ok=True)
ARQUIVO_SAIDA = "data/processed/dataset_axia_completo_2d.h5"

print("A ler dados do Excel...")
abas_excel = pd.read_excel(ARQUIVO_EXCEL, sheet_name=None)

# Consolida as abas e cria a tabela dinâmica (Tags viram colunas)
lista_sensores = []
for nome_aba, df in abas_excel.items():
    if 'tag' in df.columns and 'valor' in df.columns:
        df_temp = df[['timestamp', 'tag', 'valor']].copy()
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        lista_sensores.append(df_temp)

df_total = pd.concat(lista_sensores)
df_sensores = df_total.pivot_table(index='timestamp', columns='tag', values='valor').sort_index()

arquivos_matrizes = sorted(glob.glob(os.path.join(PASTA_MATRIZES, "*.csv")))
print(f"Encontradas {len(arquivos_matrizes)} matrizes na pasta {PASTA_MATRIZES}")

# --- PROCESSAMENTO OTIMIZADO PARA HDF5 (FORMATO 2D) ---
num_amostras = min(len(arquivos_matrizes), len(df_sensores))
dataset_metadata = []

print(f"A iniciar a criação do ficheiro HDF5 em {ARQUIVO_SAIDA}...")
with h5py.File(ARQUIVO_SAIDA, 'w') as hf:
    # Criação do dataset com a forma (Amostras, Altura, Largura)
    # Assumindo a resolução padrão de 640x480 (480 linhas e 640 colunas)
    dataset_matrizes = hf.create_dataset('matrizes_termicas', shape=(num_amostras, 480, 640), dtype=np.float32)

    for i in range(num_amostras):
        linha_sensor = df_sensores.iloc[i].to_dict()
        timestamp_excel = df_sensores.index[i]
        caminho_csv = arquivos_matrizes[i]
        nome_arquivo = os.path.basename(caminho_csv)

        # 1. Lê a matriz do CSV (Sem utilizar o flatten!)
        # O resultado já será um array numpy 2D com a forma (480, 640)
        matriz_2d = pd.read_csv(caminho_csv, header=None).values

        # Guarda a matriz diretamente na sua posição no ficheiro HDF5
        dataset_matrizes[i] = matriz_2d

        # 2. Processa os Metadados / Sensores
        timestamp_matriz = "_".join(nome_arquivo.split('_')[2:4])

        registro = {
            'id_amostra': i,
            'timestamp_matriz': timestamp_matriz,
            'timestamp_excel_original': timestamp_excel,
            'arquivo_origem': nome_arquivo
        }
        registro.update(linha_sensor)
        dataset_metadata.append(registro)

        if (i + 1) % 10 == 0:
            print(f"Processado: {i + 1}/{num_amostras}")

# 3. Exportação dos Metadados
print("A guardar os metadados dos sensores no ficheiro HDF5...")
df_meta = pd.DataFrame(dataset_metadata)

# Guarda o DataFrame no mesmo ficheiro HDF5
df_meta.to_hdf(ARQUIVO_SAIDA, key='sensores', mode='a')

print("Concluído com sucesso!")