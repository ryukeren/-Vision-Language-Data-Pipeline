# 👁️‍🗨️ Vision-Language Data Engineering Pipeline

An enterprise-grade, highly resilient data engineering pipeline designed to ingest, process, and extract structured data from unstructured visual assets (images, scanned documents) using Vision Large Language Models (LLMs).

## 🚀 Overview & The Problem It Solves

**The Challenge:** Modern enterprises are drowning in unstructured visual data—scanned invoices, handwritten forms, complex diagrams, and unformatted PDFs. Traditional OCR (Optical Character Recognition) systems are rigid, require constant template maintenance, and fail catastrophically when encountering unexpected layouts. Translating this unstructured visual data into clean, structured, queryable databases remains a massive operational bottleneck.

**The Solution:** This pipeline leverages the reasoning capabilities of state-of-the-art Vision LLMs (Google Gemini) to dynamically understand, parse, and extract structured JSON from unstructured visual inputs. It eliminates the need for rigid OCR templates, turning chaotic raw images into pristine, queryable datasets ready for downstream analytics and automation.

## 💼 Key Executive Insights & Business Value

- **Automated Data Entry at Scale:** Drastically reduces manual data entry costs and human error by intelligently automating the extraction of complex fields from diverse document types.
- **Extreme Fault Tolerance:** Built for enterprise reliability. Intermittent API failures and rate limits are handled gracefully through robust exponential backoff strategies, ensuring zero data loss during high-volume processing.
- **Real-Time Observability:** Complete transparency into the pipeline's health. Every extraction, API call, and system event is tracked, traced, and logged for auditing and performance monitoring.
- **Scalable Cloud Storage:** Seamlessly integrates with cloud-native PostgreSQL (Supabase), providing highly available, secure, and scalable storage for both raw assets and extracted structured data.

## 🧠 Methodology & AI Workflow

Our architecture is built on a resilient, agentic workflow designed for maximum reliability and precision:

1. **Agentic Orchestration (LangGraph):** The core pipeline logic is driven by LangGraph, enabling stateful, multi-step reasoning. It manages the complex flow of data from ingestion to database insertion, allowing for conditional routing and dynamic error recovery.
2. **Advanced Vision Extraction (Google Gemini):** We utilize the multimodal capabilities of Google's Gemini Vision model. Rather than just recognizing text, Gemini *understands* the context of the visual layout, allowing us to enforce strict JSON schemas for the extracted data, regardless of the input format.
3. **Resiliency Layer (Tenacity):** Enterprise APIs fail. To combat this, the pipeline wraps critical external calls (like the Gemini API and Supabase operations) in a `Tenacity` exponential backoff layer. This ensures that rate limits (HTTP 429) and transient network errors are automatically retried with increasing delays, maximizing throughput and system stability.

## 🧑‍💻 Human-in-the-Loop (HITL) Experience

Automation should empower, not operate blindly. We employ a rigorous Human-in-the-Loop (HITL) methodology to guarantee data integrity:

- **The Happy Path:** When the Vision LLM successfully extracts data that passes all schema validations and confidence thresholds, it is automatically routed directly to the database.
- **The Exception Queue:** If an anomaly is detected—such as a missing critical field, a schema validation failure, or a persistent API timeout—the payload is flagged and gracefully diverted to a dedicated HITL Review Queue.
- **Human Validation:** Domain experts can monitor the Review Queue via the frontend observability console, manually validating anomalies, correcting extraction errors, and approving records for final database ingestion. This ensures that edge cases train the system rather than breaking it.

## 🏗️ Architecture

```
vision-language-pipeline/
├── backend/          # Python 3.11+ FastAPI + LangGraph backend
├── frontend/         # React + Vite + TypeScript observability console
└── supabase/         # Database migrations, RLS policies, and seed data
```

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python 3.11+, FastAPI, LangGraph, Pydantic V2 |
| **Frontend** | React 18, Vite, TypeScript, Tailwind CSS, Axios |
| **Database** | Supabase Cloud (PostgreSQL 15) |
| **Resilience** | Tenacity (Exponential Backoff) |
| **Observability** | LangSmith (configured with APAC regional endpoint routing for real-time trace tracking) |
| **Vision LLM** | Google Gemini API |

## 🚀 Quick Start (Local Development)

If you have just cloned this repository or are starting your PC, follow these steps to run the pipeline locally.

### Prerequisites

1. **Python 3.10+** installed
2. **Node.js 18+** installed
3. **Supabase Cloud Project** (or Local Supabase)
4. **Google Gemini API Key**

### 1. Backend Setup (Terminal 1)

Open a terminal and navigate to the `backend` folder:

```bash
cd backend
```

**First time setup (after cloning):**
```bash
python -m venv .venv

# Activate the virtual environment
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create your environment variables file
cp .env.example .env
```

**⚠️ Important:** You MUST fill out the `.env` file in the `backend/` directory with your real credentials before running the server:
- `GEMINI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `DATABASE_URL` (Use the IPv4 Transaction Pooler URL for Supabase Cloud)

**Start the Backend Server (Every time you restart):**
Make sure your virtual environment is activated, then run:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*The backend is now live at `http://localhost:8000` (API Docs at `/docs`)*

### 2. Frontend Setup (Terminal 2)

Open a *second* terminal and navigate to the frontend folder:

```bash
cd frontend/observability-console
```

**First time setup (after cloning):**
```bash
npm install
```

**Start the Frontend Server (Every time you restart):**
```bash
npm run dev
```
*The frontend is now live at `http://localhost:5173`*

### 3. Database Migrations

```bash
cd supabase
supabase init          # Already done
supabase db push       # Push migrations to your remote Supabase project
# or
supabase start         # Run locally with Docker
supabase migration up  # Apply pending migrations
```

## 🗄️ Database Schema

The pipeline relies on two primary tables deployed to Supabase Cloud to manage the extraction lifecycle and human review process.

### `public.vlp_extractions`
The primary table tracking the status and payload of every file processed by the pipeline.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID (PK) | Auto-generated primary key |
| `created_at` | TIMESTAMPTZ | Audit timestamp of ingestion |
| `file_name` | TEXT | Original name of the uploaded visual asset |
| `extracted_json` | JSONB | The structured data extracted by the Vision LLM |
| `pipeline_status` | TEXT | E.g., `success`, `failed`, `needs_review` |
| `error_log` | TEXT | Detailed stack traces or API error messages |
| `execution_time_ms`| INTEGER | Total time taken by the LLM and extraction pipeline |

### `public.hitl_review_queue`
Manages the workflow for anomalies and extractions flagged for human intervention.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID (PK) | Auto-generated primary key |
| `extraction_id` | UUID (FK) | References `vlp_extractions.id` |
| `created_at` | TIMESTAMPTZ | Audit timestamp when flagged for review |
| `reviewed_status` | TEXT | Current queue state (e.g., `pending`, `approved`, `rejected`) |
| `reviewer_notes` | TEXT | Manual annotations or corrections provided by the human operator |

## 🛡️ Security

- Row Level Security (RLS) is enabled on all tables.
- Authenticated users have read-only access.
- Backend service communicates via `service_role` key with full access.
- The `service_role` key **must never** be exposed to the frontend.

## 🔐 Environment Variables

See `backend/.env.example` for required variables.
