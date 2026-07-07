# Vision-Language Data Engineering Pipeline

An enterprise-grade monorepo for a Vision-Language Data Engineering Pipeline.

## Architecture

```
vision-language-pipeline/
├── backend/          # Python 3.11+ FastAPI + LangGraph backend
├── frontend/         # React + Vite + TypeScript observability console
└── supabase/         # Database migrations, RLS policies, and seed data
```

## Tech Stack

| Layer     | Technology                                        |
|-----------|---------------------------------------------------|
| Backend   | Python 3.11+, FastAPI, LangGraph, Pydantic V2     |
| Frontend  | React 18, Vite, TypeScript, Tailwind CSS, Axios   |
| Database  | Supabase (PostgreSQL 15 + pgvector extension)     |
| Tracing   | LangSmith                                         |

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

## Database Schema

### `public.documents`
Tracks uploaded files through the pipeline.

| Column       | Type        | Description                      |
|-------------|-------------|----------------------------------|
| `id`          | UUID (PK)   | Auto-generated primary key       |
| `filename`    | TEXT        | Original filename                |
| `file_url`    | TEXT        | Storage URL                      |
| `status`      | TEXT        | `pending`, `processing`, `done`, `error` |
| `created_at`  | TIMESTAMPTZ | Audit timestamp                  |
| `updated_at`  | TIMESTAMPTZ | Last update timestamp            |

### `public.parsed_elements`
Stores structured elements extracted from documents.

| Column            | Type          | Description                           |
|------------------|---------------|---------------------------------------|
| `id`              | UUID (PK)     | Auto-generated primary key            |
| `document_id`     | UUID (FK)     | References `documents.id`             |
| `element_type`    | TEXT          | `title`, `text`, `image`, `table`, etc.|
| `content`         | TEXT          | Extracted text or caption             |
| `bounding_box`    | JSONB         | Pixel coordinates `{x, y, w, h}`      |
| `confidence_score`| FLOAT         | Model confidence (0.0–1.0)            |
| `embedding`       | vector(1536)  | OpenAI text-embedding-3-small vector  |
| `metadata`        | JSONB         | Arbitrary structured metadata         |
| `created_at`      | TIMESTAMPTZ   | Audit timestamp                       |

## Security

- Row Level Security (RLS) is enabled on all tables.
- Authenticated users have read-only access.
- Backend service communicates via `service_role` key with full access.
- The `service_role` key **must never** be exposed to the frontend.

## Environment Variables

See `backend/.env.example` for required variables.
