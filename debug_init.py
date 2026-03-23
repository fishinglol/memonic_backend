import os
import sys
import asyncio
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

async def test():
    print("Step 1: Testing WhisperModel...")
    from faster_whisper import WhisperModel
    try:
        _ = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper ready.")
    except Exception as e:
        print(f"Whisper failed: {e}")

    print("Step 2: Testing SpeakerRecognition...")
    from speechbrain.inference.speaker import SpeakerRecognition
    try:
        _ = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        print("Speaker ready.")
    except Exception as e:
        print(f"Speaker failed: {e}")

    print("Step 3: Testing EncoderClassifier...")
    from speechbrain.inference.classifiers import EncoderClassifier
    try:
        _ = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
            run_opts={"device": "cpu"}
        )
        print("Emotion ready.")
    except Exception as e:
        print(f"Emotion failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
