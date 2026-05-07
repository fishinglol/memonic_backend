#!/bin/bash
SESSION="memonic"

echo "🚀 Updating and Setting up Memonic Backend..."

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

# 4. Pull models
echo "📥 Checking AI models..."
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 5. Python Dependencies
# Use the local 'env' folder if it exists
if [ -d "env" ]; then
    echo "🐍 Installing dependencies into virtual environment..."
    ./env/bin/pip install -r requirements.txt
    ./env/bin/pip install faster-whisper --upgrade
else
    echo "⚠️ Virtual environment 'env' not found. Installing to system..."
    pip install -r requirements.txt --break-system-packages
fi

# 6. Kill old session, create fresh one
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION -n "backend"

# 7. Start Unified Backend
# We run the unified backend from the root so imports work correctly.
# Both Chat and the new Audio Processing API are now served together.
echo "🚀 Starting unified backend on port 8000..."
tmux send-keys -t $SESSION:0 "export PYTHONPATH=$PYTHONPATH:. && ./env/bin/uvicorn core.main:app --host 0.0.0.0 --port 8000 --reload" C-m

echo "✅ Setup complete! Backend is now serving /api/chat and /audio."
sleep 1
tmux attach-session -t $SESSION
