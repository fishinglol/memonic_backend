import torch

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model names / paths
WHISPER_MODEL_NAME = "base"
SPEAKER_MODEL_PATH = "pretrained_models/spkrec-ecapa-voxceleb"
EMOTION_MODEL_PATH = "pretrained_models/emotion-recognition-wav2vec2-IEMOCAP"

# ChromaDB path
CHROMA_PATH = "./memonic_memory"

# Ollama model names
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL_SUMMARY = "llama3.2:3b"

# Silence watcher thresholds
SILENCE_THRESHOLD_SECONDS = 120
SILENCE_CHECK_INTERVAL = 30
MIN_WORDS_TO_SUMMARIZE = 10
