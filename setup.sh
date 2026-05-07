#!/bin/bash
SESSION="memonic"

# Ensure we are in the project root
cd "$(dirname "$0")"

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

# 5. Python Dependencies
# Using --break-system-packages as requested for Lightning AI environment
pip install speechbrain faster-whisper transformers mem0ai --upgrade --break-system-packages

# 6. Kill old session, create fresh one
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION -n "services"

# 7. Split FIRST, before sending any commands
tmux split-window -h -t $SESSION:0

# 8. Small sleep to let both panes stabilize
sleep 1

# 9. Send commands AFTER both panes exist
# Pane 0.0 (left) → AI Audio Backend on 8001
# We use PYTHONPATH=. so it can find 'core' modules for DB access
tmux send-keys -t $SESSION:0.0 "export PYTHONPATH=\$PYTHONPATH:. && uvicorn ai.api:app --host 0.0.0.0 --port 8001" C-m

# Pane 0.1 (right) → Core Chat Backend on 8000
tmux send-keys -t $SESSION:0.1 "export PYTHONPATH=\$PYTHONPATH:. && uvicorn core.main:app --host 0.0.0.0 --port 8000" C-m

echo "✅ Setup complete! Attaching to tmux..."
sleep 1
tmux attach-session -t $SESSION
