import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
# print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


english_file = "poem_english.txt"
arabic_file = "poem_arabic.txt"
turkish_file = "poem_turkish.txt"

with open(english_file, "r", encoding="utf-8") as f:
    poem_english = f.read()

with open(arabic_file, "r", encoding="utf-8") as f:
    poem_arabic = f.read()

with open(turkish_file, "r", encoding="utf-8") as f:
    poem_turkish = f.read()


# Run TTS
# Text to speech to a file
speaker = "manu" # scholz
tts.tts_to_file(text=poem_english, speaker_wav="media/input/" + speaker + ".wav", language="en", file_path="media/output/output_en.wav")
tts.tts_to_file(text=poem_arabic, speaker_wav="media/input/" + speaker + ".wav", language="ar", file_path="media/output/output_ar.wav")
tts.tts_to_file(text=poem_turkish, speaker_wav="media/input/" + speaker + ".wav", language="tr", file_path="media/output/output_tr.wav")