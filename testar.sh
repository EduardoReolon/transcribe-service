#!/bin/bash

# 2. Envia para o seu servidor local na porta 5000
echo "Enviando para o Whisper..."
curl -X POST "http://localhost:5000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "audio=@teste-PT-BR.mp3"

echo -e "\nTeste concluído."