import torch

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model names / paths
WHISPER_MODEL_NAME = "small"
SPEAKER_MODEL_PATH = "pretrained_models/spkrec-ecapa-voxceleb"
EMOTION_MODEL_PATH = "pretrained_models/emotion-recognition-wav2vec2-IEMOCAP"

# ChromaDB path
CHROMA_PATH = "./memonic_memory"

# Silence watcher thresholds
SILENCE_THRESHOLD_SECONDS = 120
SILENCE_CHECK_INTERVAL = 30
MIN_WORDS_TO_SUMMARIZE = 10
