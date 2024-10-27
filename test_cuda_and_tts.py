import torch
from TTS.api import TTS

# Check if CUDA is installed
if torch.cuda.is_available():
    print("CUDA installed successfully\n") 
else:
    print("CUDA not properly installed. Stopping process...")
    quit()

# Print available TTS models
view_models = input("View models? [y/n]\n")
if view_models == "y":
    tts_manager = TTS().list_models()
    all_models = tts_manager.list_models()
    print("TTS models:\n", all_models, "\n", sep = "")
