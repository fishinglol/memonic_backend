print("Importing faster_whisper...")
from faster_whisper import WhisperModel
print("Import successful. Initializing model...")
model = WhisperModel("small", device="cpu", compute_type="float32")
print("Initialization successful.")
