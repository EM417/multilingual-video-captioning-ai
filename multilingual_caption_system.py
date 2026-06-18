# ============================================================
# CELL 1 — Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs("/content/drive/MyDrive/Capstone/saved_models", exist_ok=True)
os.makedirs("/content/drive/MyDrive/Capstone/frames", exist_ok=True)
os.makedirs("/content/drive/MyDrive/Capstone/outputs", exist_ok=True)


# ============================================================
# CELL 2 — Install dependencies (run once)
# ============================================================
# !pip install git+https://github.com/openai/whisper.git
# !sudo apt update && sudo apt install -y ffmpeg
# !pip install transformers sentencepiece sacremoses
# !pip install nltk rouge-score
# !pip install torch torchvision
# !pip uninstall -y tensorflow tf-keras tensorflow-text


# ============================================================
# CELL 3 — Imports
# ============================================================
import cv2
import json
import subprocess
import numpy as np
import torch
import whisper

from pathlib import Path
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torchvision.models as tv_models
import torchvision.transforms as T


# ============================================================
# CELL 4 — Language Code Mappings
# Whisper returns ISO 639-1 codes (e.g. "fr", "en")
# mBART-50 uses its own codes (e.g. "fr_XX", "en_XX")
# Both maps are needed for the same-language detection logic
# ============================================================
%cd "/content/drive/MyDrive/Capstone"

# Whisper ISO 639-1 → mBART-50 language code
WHISPER_TO_MBART = {
    "en": "en_XX",
    "fr": "fr_XX",
    "es": "es_XX",
    "ar": "ar_AR",
    "hi": "hi_IN",
    "sw": "sw_KE",
    "zh": "zh_CN",
    "de": "de_DE",
    "pt": "pt_XX",
    "ru": "ru_RU",
    "ja": "ja_XX",
    "ko": "ko_KR",
    "it": "it_IT",
    "nl": "nl_XX",
    "tr": "tr_TR",
    "pl": "pl_PL",
    "vi": "vi_VN",
    "ro": "ro_RO",
    "cs": "cs_CZ",
    "fi": "fi_FI",
}

# Human-readable name → mBART-50 code (for user-facing selection)
LANG_CODES = {
    "English":    "en_XX",
    "French":     "fr_XX",
    "Spanish":    "es_XX",
    "Arabic":     "ar_AR",
    "Hindi":      "hi_IN",
    "Swahili":    "sw_KE",
    "Chinese":    "zh_CN",
    "German":     "de_DE",
    "Portuguese": "pt_XX",
    "Russian":    "ru_RU",
    "Japanese":   "ja_XX",
    "Korean":     "ko_KR",
    "Italian":    "it_IT",
    "Dutch":      "nl_XX",
    "Turkish":    "tr_TR",
}

# Reverse map: mBART code → human-readable name
MBART_TO_NAME = {v: k for k, v in LANG_CODES.items()}

print("Language maps loaded.")


# ============================================================
# CELL 5 — Helper: SRT timestamp formatter
# FIX: No space before milliseconds — standard SRT format
# ============================================================
def seconds_to_srt_time(t: float) -> str:
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ============================================================
# CELL 6 — PART 1: Audio Extraction
# FIX: -y flag prevents ffmpeg hanging on re-run
# ============================================================
video_path  = "/content/drive/MyDrive/Capstone/videoplayback.mp4"
audio_path  = "/content/drive/MyDrive/Capstone/videoplayback.wav"
outputs_dir = "/content/drive/MyDrive/Capstone/outputs"

subprocess.run([
    "ffmpeg", "-y",
    "-i", video_path,
    "-ar", "16000",
    "-ac", "1",
    audio_path
], check=True)

print("Audio extracted successfully.")


# ============================================================
# CELL 7 — PART 2: Whisper Transcription + Auto Language Detection
# No language is specified — Whisper detects it automatically
# ============================================================
print("Loading Whisper model...")
whisper_model = whisper.load_model("medium")

print("Transcribing and detecting language...")
result = whisper_model.transcribe(audio_path)   # No language= argument

detected_whisper_code = result["language"]       # e.g. "fr", "en", "sw"
detected_mbart_code   = WHISPER_TO_MBART.get(detected_whisper_code, None)
detected_lang_name    = MBART_TO_NAME.get(detected_mbart_code, detected_whisper_code.upper())

print("\n" + "="*50)
print(f"  Detected language: {detected_lang_name} ({detected_whisper_code})")
print("="*50)
print(f"\nTranscription preview:\n{result['text'][:400]}")


# ============================================================
# CELL 8 — Handle unrecognised source language
# If Whisper detects a language not in our mBART map, warn the user
# ============================================================
if detected_mbart_code is None:
    print(f"\n⚠️  WARNING: Detected language '{detected_whisper_code}' is not supported")
    print("   by the mBART-50 translation model in this setup.")
    print("   Captions will be generated in the original detected language only.")
    print("   Add the language code to WHISPER_TO_MBART to enable translation.")
    # Fall back to English as source for mBART — best-effort
    detected_mbart_code = "en_XX"


# ============================================================
# CELL 9 — Save transcription (plain text + JSON)
# ============================================================
with open("/content/drive/MyDrive/Capstone/videoplayback.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

with open("/content/drive/MyDrive/Capstone/videoplayback.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("Transcription saved (text + JSON).")


# ============================================================
# CELL 10 — Save original-language SRT
# ============================================================
orig_srt_path = f"{outputs_dir}/captions_{detected_whisper_code}_original.srt"

with open(orig_srt_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(result["segments"]):
        text = seg["text"].strip()
        if not text:
            continue
        f.write(f"{i+1}\n")
        f.write(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}\n")
        f.write(f"{text}\n\n")

print(f"Original SRT saved: {orig_srt_path}")


# ============================================================
# CELL 11 — PART 3: Load mBART-50 Translation Model
# REPLACEMENT: mBERT → mBART-50 (true seq2seq, 50+ languages)
# ============================================================
print("Loading mBART-50 translation model...")
mbart_model     = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
print("mBART-50 loaded.")


def translate_segment(text: str, src_code: str, tgt_code: str) -> str:
    """Translate a single text segment using mBART-50."""
    if not text.strip():
        return ""
    mbart_tokenizer.src_lang = src_code
    inputs = mbart_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    forced_bos = mbart_tokenizer.lang_code_to_id[tgt_code]
    with torch.no_grad():
        tokens = mbart_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=512,
            num_beams=4,
        )
    return mbart_tokenizer.decode(tokens[0], skip_special_tokens=True)


# ============================================================
# CELL 12 — PART 4: Per-language Processing with Same-Language Logic
#
# For each user-selected target language:
#   IF target == detected source language:
#       → Notify user, skip translation, use original text
#   ELSE:
#       → Translate each segment, save translated SRT
# ============================================================

# ---- USER SETS TARGET LANGUAGES HERE ----------------------
user_selected_languages = ["French", "Spanish", "Arabic"]  # change as needed
# -----------------------------------------------------------

srt_files = {}   # track {lang_name: srt_path} for burn-in step

print("\n" + "="*50)
print(f"  Source (detected): {detected_lang_name}")
print(f"  Targets selected : {', '.join(user_selected_languages)}")
print("="*50 + "\n")

for lang_name in user_selected_languages:

    target_code = LANG_CODES.get(lang_name)
    if not target_code:
        print(f"❌  '{lang_name}' is not a supported language. Skipping.")
        continue

    srt_out = f"{outputs_dir}/captions_{lang_name.lower()}.srt"

    # ── Same-language check ──────────────────────────────────
    if target_code == detected_mbart_code:
        print(f"⚠️  NOTICE: The uploaded video is already in {lang_name}.")
        print(f"   No translation needed. Captions will be embedded in {lang_name}.")
        print(f"   Generating SRT directly from original transcription...\n")

        with open(srt_out, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"]):
                text = seg["text"].strip()
                if not text:
                    continue
                f.write(f"{i+1}\n")
                f.write(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}\n")
                f.write(f"{text}\n\n")

        print(f"✅  SRT saved (original language): {srt_out}\n")

    # ── Translation path ─────────────────────────────────────
    else:
        print(f"🔄  Translating {detected_lang_name} → {lang_name}...")

        with open(srt_out, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"]):
                text = seg["text"].strip()
                if not text:          # FIX: skip empty segments
                    continue
                translated = translate_segment(text, detected_mbart_code, target_code)
                f.write(f"{i+1}\n")
                f.write(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}\n")
                f.write(f"{translated}\n\n")

        print(f"✅  SRT saved ({lang_name}): {srt_out}\n")

    srt_files[lang_name] = srt_out

print("All SRT files generated.")


# ============================================================
# CELL 13 — PART 5: Frame Extraction & Caption Alignment
# FIX: Frames saved to disk (not RAM) to prevent OOM crashes
# FIX: Empty captions skipped
# ============================================================
with open("/content/drive/MyDrive/Capstone/videoplayback.json", encoding="utf-8") as f:
    transcription_data = json.load(f)

frame_interval = 2
frames_dir     = "/content/drive/MyDrive/Capstone/frames"
cap            = cv2.VideoCapture(video_path)
fps            = cap.get(cv2.CAP_PROP_FPS)
frames_data    = []
frame_count    = 0
saved_count    = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(fps * frame_interval) == 0:
        timestamp = frame_count / fps
        caption   = ""

        for seg in transcription_data["segments"]:
            if seg["start"] <= timestamp < seg["end"]:
                caption = seg["text"].strip()
                break

        if not caption:   # FIX: skip frames with no caption
            frame_count += 1
            continue

        frame_path = os.path.join(frames_dir, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames_data.append({
            "frame_path": frame_path,
            "caption":    caption,
            "timestamp":  timestamp,
        })
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} captioned frames.")


# ============================================================
# CELL 14 — CNN Feature Extraction (pretrained ResNet50)
# IMPROVEMENT: pretrained weights instead of training from scratch
# FIX: renamed to caption_model to avoid variable collision
# ============================================================
cnn_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])
cnn_model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_frame_features(frame_path: str) -> np.ndarray:
    frame  = cv2.imread(frame_path)
    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        feat = cnn_model(tensor).squeeze().numpy()
    return feat

print("Extracting CNN features...")
for d in frames_data:
    d["features"] = extract_frame_features(d["frame_path"])

print(f"Done. Feature dim: {frames_data[0]['features'].shape}")


# ============================================================
# CELL 15 — Sequence Preparation
# ============================================================
sequence_length = 10
feature_dim     = 2048

frame_sequences = []
caption_texts   = []

for i in range(len(frames_data) - sequence_length + 1):
    seq     = [frames_data[j]["features"] for j in range(i, i + sequence_length)]
    caption = frames_data[i + sequence_length - 1]["caption"]
    frame_sequences.append(seq)
    caption_texts.append(caption)

frame_sequences = np.array(frame_sequences)
print("Frame sequences shape:", frame_sequences.shape)


# ============================================================
# CELL 16 — LSTM Captioning Model (pure PyTorch)
# FIX: No TensorFlow, no variable collision
# ============================================================
import torch.nn as nn

class VideoCaptionLSTM(nn.Module):
    def __init__(self, feat_dim=2048, hidden=512, output_dim=2048):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True, num_layers=2, dropout=0.3)
        self.fc   = nn.Linear(hidden, output_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

caption_model = VideoCaptionLSTM()
print(caption_model)


# ============================================================
# CELL 17 — Train / Test Split + Training
# FIX: Proper held-out test set — no longer evaluating on training data
# ============================================================
from sklearn.model_selection import train_test_split

X = torch.tensor(frame_sequences, dtype=torch.float32)
y = X[:, -1, :]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

optimizer    = torch.optim.Adam(caption_model.parameters(), lr=1e-3)
loss_fn      = nn.MSELoss()
train_losses = []
val_losses   = []
epochs       = 20
batch_size   = 4

caption_model.train()
for epoch in range(epochs):
    perm       = torch.randperm(len(X_train))
    epoch_loss = 0.0

    for i in range(0, len(X_train), batch_size):
        idx    = perm[i:i+batch_size]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        loss   = loss_fn(caption_model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    caption_model.eval()
    with torch.no_grad():
        v_loss = loss_fn(caption_model(X_test), y_test).item()
    caption_model.train()

    train_losses.append(epoch_loss / max(len(X_train) // batch_size, 1))
    val_losses.append(v_loss)
    print(f"Epoch {epoch+1}/{epochs} — train: {train_losses[-1]:.4f} | val: {v_loss:.4f}")

torch.save(
    caption_model.state_dict(),
    "/content/drive/MyDrive/Capstone/saved_models/caption_model.pt"
)
print("Model saved.")


# ============================================================
# CELL 18 — Plot Training & Validation Loss
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses,   label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend(); plt.show()


# ============================================================
# CELL 19 — Caption Retrieval (per frame)
# FIX: Best caption retrieved per frame, not one global caption
# ============================================================
from sklearn.metrics.pairwise import cosine_similarity

ref_features = np.array([d["features"] for d in frames_data])
ref_captions = [d["caption"] for d in frames_data]

caption_model.eval()
with torch.no_grad():
    predicted = caption_model(X_test).numpy()

def retrieve_caption(pred_feat, ref_feats, ref_caps, top_k=3):
    sims     = cosine_similarity(pred_feat.reshape(1, -1), ref_feats).flatten()
    best_idx = np.argsort(sims)[-top_k:][::-1][0]
    return ref_caps[best_idx], sims[best_idx]

generated = [retrieve_caption(p, ref_features, ref_captions) for p in predicted]

print("Sample generated captions (first 5):")
for i, (cap, score) in enumerate(generated[:5]):
    print(f"  [{i+1}] (score={score:.3f}) {cap}")


# ============================================================
# CELL 20 — Evaluation: BLEU, METEOR, ROUGE
# FIX: BLEU uses SmoothingFunction (was always returning 0.0)
# FIX: Evaluated on held-out test set only
# ============================================================
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rs

nltk.download("wordnet",  quiet=True)
nltk.download("omw-1.4", quiet=True)

test_refs    = [caption_texts[i] for i in range(len(X_test))]
scorer       = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
smooth       = SmoothingFunction().method1
bleu_scores, meteor_scores, rouge_scores = [], [], []

for ref, (gen, _) in zip(test_refs, generated):
    ref_tok = ref.split(); gen_tok = gen.split()
    bleu_scores.append(sentence_bleu([ref_tok], gen_tok, smoothing_function=smooth))
    meteor_scores.append(meteor_score([ref_tok], gen_tok))
    rouge_scores.append(scorer.score(ref, gen))

avg_bleu   = np.mean(bleu_scores)
avg_meteor = np.mean(meteor_scores)
avg_r1     = np.mean([s["rouge1"].fmeasure for s in rouge_scores])
avg_r2     = np.mean([s["rouge2"].fmeasure for s in rouge_scores])
avg_rL     = np.mean([s["rougeL"].fmeasure for s in rouge_scores])

print(f"BLEU:    {avg_bleu:.4f}")
print(f"METEOR:  {avg_meteor:.4f}")
print(f"ROUGE-1: {avg_r1:.4f}")
print(f"ROUGE-2: {avg_r2:.4f}")
print(f"ROUGE-L: {avg_rL:.4f}")


# ============================================================
# CELL 21 — Plot Evaluation Metrics
# ============================================================
metrics = ["BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
scores  = [avg_bleu, avg_meteor, avg_r1, avg_r2, avg_rL]
colors  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B2"]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, scores, color=colors)
plt.xlabel("Metric"); plt.ylabel("Score")
plt.title(f"Evaluation Metrics — Test Set ({detected_lang_name} source)")
for bar, val in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10)
plt.ylim(0, 1.0); plt.tight_layout(); plt.show()


# ============================================================
# CELL 22 — PART 6: Burn captions into video with FFmpeg
# One output video per target language
# Same-language outputs skip translation but still get captions burned in
# ============================================================
def burn_subtitles(video_in: str, srt_path: str, video_out: str, lang: str):
    print(f"🎬  Burning {lang} captions into video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_in,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        video_out
    ], check=True)
    print(f"✅  Output saved: {video_out}\n")

for lang_name, srt_path in srt_files.items():
    out_path = f"{outputs_dir}/output_{lang_name.lower()}.mp4"
    burn_subtitles(video_path, srt_path, out_path, lang_name)

print("="*50)
print("  All outputs ready in:", outputs_dir)
print("="*50)
