#!/bin/bash

# Loop para arquivos que contêm V3.A3 e terminam em .IMA
for arquivo in *V4.A3*.IMA; do
    # Verifica se existem arquivos para evitar erros
    [ -e "$arquivo" ] || continue

    # Cria o nome de saída removendo .IMA e adicionando .jpg
    # Exemplo: ARQUIVO_TESTE.IMA vira ARQUIVO_TESTE.jpg
    nome_saida="${arquivo%.IMA}.jpg"

    echo "Processando: $arquivo"
    echo "Salvando como: $nome_saida"

    # Executa o curl
    curl -X 'POST' \
      'http://10.0.224.8:8000/api/visualize' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F "file=@$arquivo" \
      -F 'colormap=turbo' \
      --output "$nome_saida"

    echo "--------------------------------"
done
