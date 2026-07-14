# 👁️‍🗨️ Vision-Language Data Engineering Pipeline

> **Enterprise-grade, dual-mode AI platform** for automated structured data extraction from invoices and real-time video analytics — powered by Google Gemini 2.5 Flash, LangGraph, FastAPI, and React.

---

## 🚀 Overview & The Problem It Solves

**The Challenge:** Modern enterprises are drowning in unstructured visual data — scanned invoices, handwritten forms, and hours of video footage. Traditional OCR systems are rigid, require constant template maintenance, and fail on unexpected layouts. Meanwhile, video analytics platforms are either cloud-black-boxes or GPU-hungry on-premise deployments that most teams can't run locally.

**The Solution:** This platform provides **two production-grade AI pipelines** through a single unified console:

1. **Invoice Extraction Pipeline** — A LangGraph agentic pipeline that ingests document images and extracts fully structured, schema-validated JSON using Gemini's vision reasoning — no templates, no rigid OCR rules.
2. **Video Analytics Tracker** — A Hybrid Cloud-Edge architecture that accepts `.mp4` uploads, streams them to the Gemini Files API for frame-level understanding, and returns a typed `VideoTrackerReport` with event detection, object labels, timestamps, and bounding boxes.

Both pipelines are accessible from a **single React observability console** with tab navigation, custom analysis prompts, and real-time status tracking.

---

## 💼 Key Executive Insights & Business Value

| Benefit | Detail |
|:---|:---|
| 🤖 **Automated Data Entry** | Eliminates manual invoice processing. Gemini understands context, not just pixels — handles diverse layouts without template maintenance. |
| ⚡ **Extreme Fault Tolerance** | Self-healing LangGraph graph with up to 3 retry attempts per job. Tenacity exponential backoff on all Gemini API calls. Zero data loss during transient failures. |
| 🌐 **Hybrid Cloud-Edge** | No local GPU required. FastAPI handles upload streaming and temp-file management; Gemini 2.5 Flash handles inference; local Pydantic validates the response. |
| 📊 **Real-Time Observability** | Every extraction, retry, and API event is logged and surfaced in the frontend. LangSmith trace integration for deep pipeline debugging. |
| 🗄️ **Scalable Storage** | Cloud-native PostgreSQL (Supabase) stores raw assets and extracted structured data with Row Level Security enforced at the database layer. |
| 🎯 **Custom Prompt Control** | Both pipelines expose a user-editable **Analysis Prompt** field. Users can focus Gemini's attention on specific fields, objects, or events without touching code. |

---

## 🧠 Methodology & AI Workflow

### Invoice Extraction Pipeline (Port 8000)

```
Image Upload (multipart/form-data)
        ↓
FastAPI /api/v1/upload   [file + custom prompt as Form fields]
        ↓
LangGraph Orchestrator
  ├─ Classification Node  ── Guardrail: rejects non-invoice files
  ├─ Extraction Node      ── Gemini 2.5 Flash + response_schema enforcement
  ├─ Validation Node      ── Local Pydantic ComprehensiveInvoiceSchema check
  └─ Self-Healing Node    ── On failure: injects Pydantic error into retry prompt
        ↓
Supabase Cloud PostgreSQL  (vlp_extractions table)
        ↓
Frontend polls /api/v1/status/{doc_id} for live telemetry
```

### Video Analytics Tracker (Port 8001)

```
MP4 Upload (multipart/form-data)
        ↓
FastAPI /api/v1/analyze-video   [shutil.copyfileobj streaming — no RAM spikes]
        ↓
Gemini Files API upload         [video/mp4, polled until ACTIVE]
        ↓
gemini-2.5-flash inference      [response_schema=VideoTrackerReport enforced cloud-side]
        ↓
Local Pydantic validation       [VideoTrackerReport double-check]
        ↓
Gemini cloud file deleted       [quota cleanup in finally block]
Local temp file deleted         [always, success or failure]
        ↓
Structured JSON response → React UI
```

---

## 🧑‍💻 Human-in-the-Loop (HITL) Experience

Automation should empower, not operate blindly. The invoice pipeline employs a rigorous HITL methodology to guarantee data integrity:

- **Happy Path:** When Gemini extracts data that passes all schema validations, it is automatically committed to Supabase.
- **Exception Queue:** Schema failures or persistent API errors are flagged and routed to the `hitl_review_queue` table.
- **Self-Healing:** On validation failure, the exact Pydantic error is injected back into the next prompt so Gemini can correct its own output — up to `max_retries` (default: 2) times.
- **Custom Prompt Override:** Users can provide a domain-specific extraction directive (e.g., *"Focus on VAT line items and payment due dates"*) to guide Gemini without modifying code.

---

## 🏗️ Architecture

```
vision-language-pipeline/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI gateway — Invoice pipeline (Port 8000)
│   │   ├── schemas/
│   │   │   └── extraction.py        # ComprehensiveInvoiceSchema (Pydantic)
│   │   └── services/
│   │       ├── orchestrator.py      # LangGraph state graph + PipelineState TypedDict
│   │       └── real_vlm.py          # live_vlm_extract() — Gemini vision call + retry
│   │
│   ├── server.py                    # FastAPI gateway — Video Analytics (Port 8001)
│   ├── qwen_structured.py           # VideoTrackerReport schema + JSON cleaner
│   ├── qwen_inference.py            # Local Qwen2.5-VL engine (research/offline use)
│   ├── config.py                    # Pydantic Settings — reads .env
│   ├── db.py                        # Supabase cloud client
│   └── requirements.txt
│
├── frontend/
│   └── observability-console/
│       └── src/
│           ├── App.tsx              # Invoice console + tab navigation
│           └── components/
│               └── VideoAnalyzer.tsx # Video Analytics Tracker UI
│
└── supabase/
    └── migrations/                  # PostgreSQL schema migrations + RLS policies
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Invoice Backend** | Python 3.10+, FastAPI, LangGraph, Pydantic V2, Tenacity |
| **Video Backend** | FastAPI, Pydantic V2, Google Gemini Files API |
| **Inference** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| **Frontend** | React 18, Vite, TypeScript, Tailwind CSS, Lucide React |
| **Database** | Supabase Cloud (PostgreSQL 15) |
| **Resilience** | Tenacity exponential backoff (2s → 4s → 8s → 15s cap, 5 attempts) |
| **Observability** | LangSmith (APAC regional endpoint, real-time trace tracking) |
| **Schema Enforcement** | Pydantic V2 (local) + Gemini `response_schema` (cloud-side) |

---

## 📦 Build Phases

This project was developed in four incremental phases:

| Phase | Deliverable | Key Files |
|:---|:---|:---|
| **Phase 1** | Core local inference engine research (Qwen2.5-VL) | `qwen_inference.py` |
| **Phase 2** | Structured output layer — Pydantic schema + JSON cleaner | `qwen_structured.py` |
| **Phase 3** | Hybrid Cloud-Edge API — FastAPI + Gemini Files API | `server.py` |
| **Phase 4** | Dual-tab React frontend — Invoice Console + Video Tracker | `App.tsx`, `VideoAnalyzer.tsx` |

---

## 🚀 Quick Start (Local Development)

### Prerequisites

1. **Python 3.10+**
2. **Node.js 18+**
3. **Supabase Cloud Project** (free tier works)
4. **Google Gemini API Key** (free tier: `https://aistudio.google.com/`)

---

### 1. Backend — Invoice Pipeline (Terminal 1, Port 8000)

```bash
cd backend

# First time only: create virtual environment and install dependencies
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Fill out .env with your credentials (see Environment Variables section below)
```

**Start the server:**
```bash
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
> API Docs: `http://localhost:8000/docs`

---

### 2. Backend — Video Analytics Tracker (Terminal 2, Port 8001)

With the same venv activated:

```bash
cd backend
.\.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8001
```

> Health check: `http://localhost:8001/health`
> Expected: `{"status":"healthy","architecture":"hybrid-cloud-edge","local_gpu_required":false}`

---

### 3. Frontend (Terminal 3)

```bash
cd frontend/observability-console

# First time only
npm install

# Every time
npm run dev
```

> Open: `http://localhost:5173`
> - **Invoice Pipeline tab** → drag & drop an invoice image → edit the Analysis Prompt → click **Trigger VLM Pipeline**
> - **Video Tracker tab** → drag & drop an `.mp4` file → preview it inline → click **Analyze Video**

---

### 4. Database Migrations

```bash
cd supabase
supabase db push       # Push migrations to your remote Supabase project
# or
supabase start         # Run locally with Docker
supabase migration up  # Apply pending migrations
```

---

## 🔐 Environment Variables

Create `backend/.env` from `backend/.env.example`:

```env
# Google Gemini — used by BOTH pipelines
GEMINI_API_KEY=your-gemini-api-key-here

# Supabase — used by BOTH pipelines for data persistence
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres:[password]@[host]:6543/postgres

# LangSmith Observability (optional but recommended)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=vision-language-pipeline
```

> ⚠️ **Never expose `SUPABASE_SERVICE_ROLE_KEY` to the frontend.** It bypasses all Row Level Security.

---

## 🗄️ Database Schema

### `public.vlp_extractions`
Tracks the status and payload of every file (invoice or video) processed by the pipelines.

| Column | Type | Description |
|:---|:---|:---|
| `id` | UUID (PK) | Auto-generated primary key |
| `created_at` | TIMESTAMPTZ | Audit timestamp of ingestion |
| `document_id` | TEXT | Unique identifier for the job/file |
| `job_type` | TEXT | `invoice` or `video` |
| `prompt_used` | TEXT | The custom analysis prompt supplied by the user |
| `status` | TEXT | `completed`, `failed`, `rejected`, etc. |
| `parsed_data` | JSONB | Structured Pydantic data extracted by Gemini |
| `error_message` | TEXT | Detailed error messages or Pydantic validation failures |
| `retry_count` | INTEGER | Number of self-healing retry attempts used |
| `raw_vlm_output` | TEXT | The raw unparsed text response from Gemini |

### `public.hitl_review_queue`
Manages the workflow for anomalies flagged for human intervention.

| Column | Type | Description |
|:---|:---|:---|
| `id` | UUID (PK) | Auto-generated primary key |
| `extraction_id` | UUID (FK) | References `vlp_extractions.id` |
| `created_at` | TIMESTAMPTZ | Timestamp when flagged for review |
| `reviewed_status` | TEXT | `pending`, `approved`, `rejected` |
| `reviewer_notes` | TEXT | Manual annotations from the human operator |

---

## 📋 API Reference

### Invoice Pipeline (Port 8000)

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/upload` | Upload invoice image + optional `prompt` (Form fields) |
| `GET` | `/api/v1/status/{document_id}` | Poll job status and extracted data |

### Video Analytics Tracker (Port 8001)

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check + client readiness |
| `POST` | `/api/v1/analyze-video` | Upload `.mp4` + optional `prompt` — returns `VideoTrackerReport` |

### `VideoTrackerReport` Schema

```json
{
  "event_detected": true,
  "summary": "A person walks across the frame and picks up an object.",
  "tracked_objects": [
    {
      "label": "person",
      "timestamp_seconds": 3.5,
      "box_2d": [0.12, 0.08, 0.85, 0.42]
    }
  ]
}
```

---

## 🛡️ Security

- Row Level Security (RLS) is enabled on all Supabase tables.
- Authenticated users have read-only access; backend communicates via `service_role` key.
- All uploaded files are written to isolated temp paths and **deleted immediately** after processing (both locally and from Gemini cloud storage).
- The `GEMINI_API_KEY` and `SUPABASE_SERVICE_ROLE_KEY` are server-side only — never sent to the browser.

---

## 🔮 Roadmap

- [x] Persist `VideoTrackerReport` results to Supabase alongside invoice extractions
- [ ] Add video thumbnail preview to the tracker results panel
- [x] Integrate bounding box overlay renderer on video frames *(Note: Implemented in Phase 5, but reverted by user preference for cleaner UX)*
- [ ] Expand HITL queue UI to the frontend observability console
- [ ] Add batch upload mode for invoice processing queues
- [ ] Docker Compose for one-command full-stack startup
