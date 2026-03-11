#!/bin/bash

URL="http://10.0.224.8:8000/api/stats"
OUTPUT="output.csv"

# Cria o cabeçalho do CSV
echo "width,height,min_temp,max_temp,mean_temp,std_temp,file_type,filename" > "$OUTPUT"

# Itera sobre os arquivos que correspondem ao padrão
for file in *V4.A3*.IMA; do
    # Verifica se o arquivo existe para evitar erros caso não haja correspondência
    if [ -e "$file" ]; then
        echo "Processando: $file"
        
        # Envia o arquivo e captura a resposta JSON silenciando o progresso do curl (-s)
        response=$(curl -s -X 'POST' "$URL" \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F "file=@$file")
        
        # Usa o jq para extrair os valores e formatar como CSV
        # Se não tiver o jq instalado: sudo apt install jq (Debian/Ubuntu) ou yum install jq (CentOS)
        parsed_data=$(echo "$response" | jq -r '[.width, .height, .min_temp, .max_temp, .mean_temp, .std_temp, .file_type] | @csv')
        
        # Salva no arquivo CSV adicionando o nome do arquivo ao final
        echo "$parsed_data,\"$file\"" >> "$OUTPUT"
    fi
done

echo "Concluído. Dados salvos em $OUTPUT"
