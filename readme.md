# 🎬 Multilingual Video Captioning AI

An end-to-end AI system that automatically transcribes speech from any video, translates it into multiple languages, and burns the translated captions directly into the output video — powered by OpenAI Whisper and mBART-50.

> **Undergraduate Capstone Project** — SRM University, Andhra Pradesh  
> Author: Eliud Mbouala Akiana

---

## 🌐 Live Demo

> Coming soon — deployment in progress

---

## 📌 What It Does

1. **Upload** any video in any language
2. **Auto-detect** the spoken language using OpenAI Whisper
3. **Select** one or more target languages from 14 supported options
4. **Translate** the transcription using Meta's mBART-50 model
5. **Embed** the translated captions permanently into the video using FFmpeg
6. **Download** the captioned video (MP4) and subtitle file (SRT) per language

### Same-Language Detection
If the user selects a target language that matches the detected source language, the system notifies the user and embeds captions without translation.

---

## 🧱 System Architecture

```
User uploads video
        ↓
FastAPI Backend (Python)
        ↓
FFmpeg → extracts audio (WAV)
        ↓
OpenAI Whisper → transcribes + auto-detects language
        ↓
mBART-50 → translates per target language
        ↓
FFmpeg → burns captions into video (MP4)
        ↓
Supabase Storage → stores output files
        ↓
React Frontend → displays download links
```

---

## 🗂️ Project Structure

```
multilingual-video-captioning-ai/
│
├── backend/                        ← FastAPI Python backend
│   ├── main.py                     ← API routes
│   ├── pipeline.py                 ← Full ML pipeline
│   ├── language_maps.py            ← Language code mappings
│   ├── utils.py                    ← SRT formatter, FFmpeg helpers
│   ├── requirements.txt            ← Python dependencies
│   └── .env                        ← Environment variables (not committed)
│
├── frontend/                       ← React frontend (Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── UploadPanel.jsx
│   │   │   ├── LanguageSelector.jsx
│   │   │   ├── ProgressTracker.jsx
│   │   │   ├── ResultsPanel.jsx
│   │   │   └── Notification.jsx
│   │   ├── hooks/
│   │   │   └── useJobStatus.js     ← Supabase Realtime + polling
│   │   ├── pages/
│   │   │   └── Home.jsx
│   │   ├── apiClient.js            ← Axios instance
│   │   ├── supabaseClient.js       ← Supabase JS client
│   │   └── App.jsx
│   ├── .env.local                  ← Frontend env variables (not committed)
│   └── package.json
│
├── video_caption_model.ipynb       ← Research notebook (Colab)
├── .gitignore
└── README.md
```

---

## 🤖 Models Used

| Model | Purpose | Size |
|---|---|---|
| **OpenAI Whisper (medium)** | Speech transcription + language detection | ~1.5GB |
| **Meta mBART-50** | Multilingual translation (50+ languages) | ~2.4GB |
| **ResNet50 (ImageNet)** | Frame feature extraction | ~100MB |
| **LSTM (custom)** | Temporal sequence modeling | ~50MB |

---

## 🌍 Supported Languages (14 Verified)

| Language | Code | Flag |
|---|---|---|
| English | en_XX | 🇬🇧 |
| French | fr_XX | 🇫🇷 |
| Spanish | es_XX | 🇪🇸 |
| Hindi | hi_IN | 🇮🇳 |
| Portuguese | pt_XX | 🇧🇷 |
| Italian | it_IT | 🇮🇹 |
| German | de_DE | 🇩🇪 |
| Dutch | nl_XX | 🇳🇱 |
| Romanian | ro_RO | 🇷🇴 |
| Polish | pl_PL | 🇵🇱 |
| Turkish | tr_TR | 🇹🇷 |
| Vietnamese | vi_VN | 🇻🇳 |
| Czech | cs_CZ | 🇨🇿 |
| Finnish | fi_FI | 🇫🇮 |

---

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| React 18 + Vite | UI framework |
| Tailwind CSS 3 | Styling |
| Axios | HTTP requests to backend |
| React Dropzone | Drag-and-drop video upload |
| Supabase JS | Realtime job updates + storage |

### Backend
| Technology | Purpose |
|---|---|
| FastAPI | Python web framework |
| Uvicorn | ASGI server |
| OpenAI Whisper | Transcription + language detection |
| mBART-50 (HuggingFace) | Multilingual translation |
| PyTorch | Model inference |
| OpenCV | Frame extraction |
| FFmpeg | Audio extraction + caption burn-in |
| Supabase Python | Storage + database client |

### Infrastructure
| Service | Purpose |
|---|---|
| Vercel | Frontend hosting |
| Render | Backend hosting |
| Supabase | Storage, database, realtime |

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10.x
- Node.js 22.x
- FFmpeg installed and in PATH
- Supabase account

### 1. Clone the repository
```bash
git clone https://github.com/EM417/multilingual-video-captioning-ai.git
cd multilingual-video-captioning-ai
```

### 2. Backend setup
```bash
cd backend
py -3.10 -m venv venv
venv\Scripts\activate        # Windows
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Create `backend/.env`:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_role_key
VIDEOS_BUCKET=videos
OUTPUTS_BUCKET=outputs
MAX_UPLOAD_SIZE_MB=30
TEMP_DIR=/tmp/caption_pipeline
```

Start the backend:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend setup
```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_API_URL=http://localhost:8000
```

Start the frontend:
```bash
npm run dev
```

### 4. Open the app
```
http://localhost:5173
```

---

## 📊 Research Notebook

The `video_caption_model.ipynb` notebook demonstrates the full ML pipeline for academic purposes:

- Audio extraction with FFmpeg
- Speech transcription with Whisper
- Multilingual translation with mBART-50
- Frame extraction with OpenCV
- CNN feature extraction with pretrained ResNet50
- Temporal modeling with a custom LSTM
- Caption retrieval using cosine similarity
- Evaluation with BLEU, METEOR, and ROUGE scores

### Key fixes applied to the original notebook

| Bug | Fix |
|---|---|
| mBERT used as generator (encoder-only) | Replaced with mBART-50 (seq2seq) |
| SRT timestamp format incorrect | Fixed `HH:MM:SS,mmm` format |
| ffmpeg hung on re-run | Added `-y` overwrite flag |
| Evaluated on training data | Proper train/test split |
| BLEU always returned 0.0 | Added SmoothingFunction |
| Raw frames stored in RAM → OOM | Frames saved to disk |
| TensorFlow + PyTorch conflict | Pure PyTorch throughout |
| language hardcoded as English | Auto-detection — no language argument |

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/languages` | List of supported languages |
| POST | `/upload` | Upload a video file |
| POST | `/process` | Start caption generation |
| GET | `/status/{job_id}` | Get current job status |

Full interactive API docs available at:
```
http://localhost:8000/docs
```

---

## 📋 Job Status Flow

```
pending → queued → extracting_audio → detecting_language
       → translating → embedding_captions → uploading → done
```

---

## ⚠️ Limitations

- Maximum upload size: **30MB** (Supabase free tier constraint)
- Processing time: **3–10 minutes** per video on CPU (no GPU)
- Audio only — the system transcribes **speech**, not visual content
- No voice dubbing — translated text appears as **on-screen captions only**
- Real-time processing not supported — videos are processed sequentially

---

## 📄 License

This project is for educational and research purposes as part of an undergraduate capstone at SRM University, Andhra Pradesh.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
- [Meta mBART-50](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) — multilingual translation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — model hub
- [Supabase](https://supabase.com) — backend infrastructure
- [FFmpeg](https://ffmpeg.org) — video processing
