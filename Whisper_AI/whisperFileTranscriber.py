import whisper
import os
import numpy as np
import torch

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using GPU to run whisper!") if torch.cuda.is_available() else print("Using CPU to run whisper!")

model = whisper.load_model("small", device=DEVICE)
print
(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

audio = whisper.load_audio("Whisper_AI/Recording (13).m4a")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)

_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

options = whisper.DecodingOptions(language='english', without_timestamps=True, fp16 = False)
result = whisper.decode(model, mel, options)
print(result.text)

#result = model.transcribe(audio = "Whisper AI/Recording (2).m4a")
#print(result["text"])