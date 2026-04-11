#!/bin/bash
SESSION="memonic"

echo "🚀 Updating and Setting up Memonic on Lightning AI..."

# 1. Update code
git pull

# 2. Check and Install Ollama if missing
if ! command -v ollama &> /dev/null; then
    echo "📦 Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 3. Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🧠 Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi

# 4. Pull models (only if needed)
echo "📥 Checking AI models..."
ollama pull llama3.2:3b
ollama pull nomic-embed-text


# 2. Python Dependencies
pip install speechbrain --upgrade --break-system-packages
pip install transformers tokenizers==0.15.2 --break-system-packages
pip install mem0ai --upgrade --break-system-packages

# 3. Kill old session, create new one
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION -n "services"
sleep 0.5  # ← give tmux time to init

# 4. Pane 0: AI Backend (Port 8001)
tmux send-keys -t $SESSION:0.0 "cd ~/memonic_backend/ai && uvicorn api:app --host 0.0.0.0 --port 8001" C-m
sleep 0.3  # ← let pane 0 settle

# 5. Split and run Core Backend (Port 8000)
tmux split-window -h -t $SESSION:0
sleep 0.3  # ← wait for new pane to exist
tmux send-keys -t $SESSION:0.1 "cd ~/memonic_backend/core && uvicorn main:app --host 0.0.0.0 --port 8000" C-m

sleep 1
echo "✅ Setup complete! Entering tmux..."
tmux attach-session -t $SESSION
