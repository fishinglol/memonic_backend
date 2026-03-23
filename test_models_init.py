import sys
import os
sys.path.append(os.path.abspath('ai'))
import models
import asyncio

async def test():
    print("Testing models.init_models()...")
    # Using defaults from ai/config.py if possible
    from ai.config import DEVICE, WHISPER_MODEL_NAME, SPEAKER_MODEL_PATH, EMOTION_MODEL_PATH
    await models.init_models(device=DEVICE,
                             whisper_name=WHISPER_MODEL_NAME,
                             speaker_path=SPEAKER_MODEL_PATH,
                             emotion_path=EMOTION_MODEL_PATH)
    print("Models initialized successfully.")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    asyncio.run(test())
