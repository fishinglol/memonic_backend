#!/bin/bash
echo "🚀 Setting up Memonic..."

curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 3
ollama pull llama3.2:1b
ollama pull nomic-embed-text

pip install speechbrain --upgrade --break-system-packages
pip install transformers tokenizers==0.15.2 --break-system-packages
pip install mem0ai --upgrade --break-system-packages

echo "✅ Done! Now run: cd ~/memonic_backend/ai && uvicorn api:app --host 0.0.0.0 --port 8000"
