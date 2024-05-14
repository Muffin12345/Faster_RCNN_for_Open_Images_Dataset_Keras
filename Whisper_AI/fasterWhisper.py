import os
import torch
from faster_whisper import WhisperModel
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


model_size = "small"

model = WhisperModel(model_size, device="cuda", compute_type="float32")
segments, info = model.transcribe("Whisper AI/Recording (6).m4a", beam_size=5, language="en", condition_on_previous_text=False)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))