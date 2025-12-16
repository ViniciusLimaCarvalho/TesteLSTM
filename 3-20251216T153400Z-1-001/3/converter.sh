#!/bin/bash

# Inicializa o contador
contador=1

# Loop para todos os arquivos terminados em .IMA
for arquivo in *.IMA; do
    # Verifica se existem arquivos para evitar erros
    [ -e "$arquivo" ] || continue

    echo "Enviando: $arquivo..."

    # Executa o curl
    # Note o uso de aspas duplas no -F para permitir a variável
    curl -X 'POST' \
      'http://10.0.224.8:8000/api/visualize' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F "file=@$arquivo" \
      -F 'colormap=turbo' \
      --output "imagem${contador}.jpg"

    echo " Salvo como imagem${contador}.jpg"
    echo "--------------------------------"

    # Incrementa o contador
    ((contador++))
done
