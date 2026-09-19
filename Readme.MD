# TheClude AI Microservice

FastAPI + LangGraph-based AI service for **TheClude** — a crypto & financial education platform. This service handles video transcription, semantic search, and AI-powered Q&A for video lessons, running independently of the main Django backend.

---

## 🏗️ Architecture

```
Django Backend (Cloudinary/S3)
        │  video_id + video_url
        ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI (this service)                  │
│                                                            │
│   POST /upload ──► PostgreSQL "queued" ──► Redis Queue    │
│   GET /status ◄── PostgreSQL (persistent job status)      │
│   POST /ask-video ──► ChromaDB search ──► Gemini answer   │
└─────────────────────────────────────────────────────────┘
        │                                    ▲
        ▼                                    │
┌───────────────────┐              ┌─────────────────────┐
│    RQ Worker       │              │      ChromaDB        │
│  (background jobs)  │──────────►  │  (vector storage)    │
│                     │  store       │                      │
│  download → ffmpeg  │              └─────────────────────┘
│  → whisper → chunk  │
│  → embed → store    │
└───────────────────┘
        │
        ▼
   Redis + PostgreSQL (Docker)
```

**Why this design:** the old Django implementation processed videos synchronously — the client waited 50+ seconds for a single upload request. Here, `/upload` returns instantly, a background worker does the heavy lifting (FFmpeg, Whisper, embeddings), and job status is tracked in PostgreSQL so it survives server restarts.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| Orchestration | LangGraph (StateGraph) |
| Background jobs | Redis + RQ |
| Job status storage | PostgreSQL (raw SQL, no ORM) |
| Vector database | ChromaDB (persistent, local) |
| Speech-to-text | faster-whisper |
| Audio extraction | FFmpeg |
| Embeddings + LLM | Google Gemini (`gemini-embedding-001`, `gemini-3.5-flash`) |
| Containerization | Docker Compose (Postgres + Redis) |

---

## 📡 API Endpoints

### Video Processing

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/v1/video/upload` | Accepts `video_id` + `video_url`, queues the video for background processing (download → extract audio → transcribe → chunk → embed → store in ChromaDB). Returns immediately with `status: "queued"`. |
| `GET` | `/api/v1/video/status/{video_id}` | Returns the current processing status (`queued` / `processing` / `completed` / `failed`), plus `chunks_stored` on success or `error_message` on failure. |

**Upload request body:**
```json
{
  "video_id": "abc123",
  "video_url": "https://res.cloudinary.com/.../lesson1.mp4"
}
```

### Video Q&A

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/v1/video/ask-video` | Asks a question about a processed video. Runs the Q&A pipeline (embed question + fetch chat history in parallel → search ChromaDB → generate answer with Gemini → save the turn to history) and returns the complete answer in one response. |
| `POST` | `/api/v1/video/ask-video-stream` | Same as above, but streams the answer back piece by piece as Gemini generates it (like ChatGPT's typing effect), instead of waiting for the full answer. Session ID is returned in the `X-Session-Id` response header. |

**Ask-video request body:**
```json
{
  "video_id": "abc123",
  "session_id": "optional — omit to start a new conversation",
  "question": "What is capitulation in crypto trading?"
}
```

**Response:**
```json
{
  "session_id": "session_xyz789",
  "answer": "Based on the video, capitulation refers to...",
  "error_message": null
}
```

A video must be `completed` (checked via job status, with a ChromaDB fallback in case the status record was lost to a server restart) before it can be queried — otherwise the route returns `404` (never submitted), `409` (still processing), or `500` (processing failed).

### System

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Health check — confirms the service is running. |
| `GET` | `/` | Root — basic welcome message and link to `/docs`. |
| `GET` | `/docs` | Interactive Swagger UI — try every endpoint directly from the browser. |

---

## 🧩 How a video gets processed (step by step)

1. **Download** — the remote video URL (Cloudinary/S3) is downloaded to a local temp file
2. **Extract audio** — FFmpeg strips the audio track into a 16kHz mono `.wav` file (the format Whisper expects)
3. **Transcribe** — faster-whisper converts speech to timestamped text segments
4. **Chunk** — segments are grouped into ~1000-character chunks, each tagged with its `[HH:MM:SS - HH:MM:SS]` timestamp range
5. **Embed** — each chunk is converted into a vector (via Gemini) in a single batched API call
6. **Store** — chunks + embeddings are persisted in ChromaDB, tagged with `video_id`
7. **Cleanup** — the temp video/audio files are deleted, whether the run succeeded or failed

Each step is a LangGraph node with conditional error routing — if any step fails, all remaining steps are skipped (never called), and the failure reason is recorded in PostgreSQL.

---

## 🚀 Local Development Setup

### 1. Start infrastructure (Postgres + Redis)
```bash
docker compose up -d
docker ps   # confirm both containers show "healthy"
```

### 2. Set up the Python environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy `.env.example` to `.env` and fill in:
```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://ai_service:ai_service@localhost:5433/ai_service
REDIS_URL=redis://localhost:6379/0
VIDEO_QUEUE_NAME=video_processing
CHROMA_DB_DIR=chroma_db
```

### 4. Run the API server
```bash
uvicorn app.main:app --reload --reload-dir app --port 8001
```

### 5. Run the background worker (separate terminal)
```bash
python -m app.worker
```

> **Windows note:** the worker uses RQ's `SimpleWorker` instead of the default `Worker`/`SpawnWorker`, because both rely on Unix-only process APIs (`os.fork()` / `os.wait4()`) that don't exist on Windows.

### 6. Try it out
Visit `http://127.0.0.1:8001/docs` for the interactive API explorer.

### 7. Testing with Postman

Once the server is running, you can test the APIs using Postman:

1. **Health Check:**
   - **Method:** `GET`
   - **URL:** `http://127.0.0.1:8001/health`
   - **Expected:** `{"status": "healthy", ...}`

2. **Process a Video:**
   - **Method:** `POST`
   - **URL:** `http://127.0.0.1:8001/api/v1/video/upload`
   - **Body (raw JSON):** `{"video_id": "test_001", "video_url": "https://www.w3schools.com/html/mov_bbb.mp4"}`
   - **Expected:** `202 Accepted` (Background task queued).

3. **Check Status:**
   - **Method:** `GET`
   - **URL:** `http://127.0.0.1:8001/api/v1/video/status/test_001`
   - **Expected:** Status `processing` or `completed`. Wait for `completed` before asking questions.

4. **Ask a Question:**
   - **Method:** `POST`
   - **URL:** `http://127.0.0.1:8001/api/v1/video/ask-video`
   - **Body (raw JSON):** `{"video_id": "test_001", "question": "What is in the video?"}`
   - **Expected:** Full JSON response with the generated `answer` and `session_id`.

5. **Streaming Chat:**
   - **Method:** `POST`
   - **URL:** `http://127.0.0.1:8001/api/v1/video/ask-video-stream`
   - **Body (raw JSON):** Same as above.
   - **Expected:** Streaming text response (ChatGPT style typing effect).

---

## 📁 Project Structure

```
ai_service/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── worker.py                  # RQ worker process
│   ├── core/
│   │   ├── config.py              # Settings (.env loader)
│   │   └── database.py            # PostgreSQL connection + schema setup
│   ├── api/routes/
│   │   ├── video.py                # /upload, /status
│   │   └── video_qa.py             # /ask-video, /ask-video-stream
│   ├── models/
│   │   ├── video.py                # Request/response schemas
│   │   └── chat.py
│   ├── nodes/                      # Individual processing steps (plain functions)
│   │   ├── video_downloader.py
│   │   ├── audio_extractor.py
│   │   ├── transcriber.py
│   │   ├── text_chunker.py
│   │   ├── embedder.py
│   │   ├── llm_answer.py
│   │   └── vector_store.py         # ChromaDB operations
│   ├── graphs/                     # LangGraph pipelines (nodes wired together)
│   │   ├── video_processing_graph.py
│   │   └── qa_graph.py
│   ├── services/
│   │   ├── job_tracker.py          # PostgreSQL-backed job status
│   │   ├── video_queue.py          # Redis/RQ enqueue
│   │   └── file_cleanup.py
│   └── tasks/
│       └── video_processing.py     # RQ task entry point
├── eval/                            # RAG evaluation (golden dataset + metrics)
├── tests/                           # Manual + unit test scripts
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## ⚠️ Known Limitations / Roadmap

- **No API Authentication for this service:** The main Django app handles user logins securely. However, this FastAPI microservice currently has no locks. In a production environment, it should be placed behind a firewall so only the Django backend can access it.
- **No Rate Limiting:** There is currently no restriction on how many times per second these specific AI endpoints can be called.
- **Old Features Not Ported Yet:** The global PDF chatbot assistant from the old Django version has not been moved to this new FastAPI service yet.
- **No Overlap in Text Chunks:** When cutting the video's transcript into small pieces for the AI to read, the pieces don't overlap. Sometimes, a sentence split exactly in the middle might lose a bit of context.
- **No Live Progress Bar for Video Processing:** When a video is processing in the background, the system does not stream the live percentage (e.g., 10%... 50%). The frontend must ask the server "Is it done yet?" (Polling). *(Note: Q&A chat answers ARE live-streamed word-by-word just like ChatGPT).*

---

## 🔗 Related

This service is called by the **TheClude** Django backend, which handles Cloudinary uploads, user auth, subscriptions, and the rest of the platform.