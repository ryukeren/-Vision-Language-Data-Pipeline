# Hybrid Cloud-Edge Vision-Language Pipeline

A production-ready video and document analytics system leveraging Gemini 2.5 Flash for cloud inference, anchored by a robust FastAPI asynchronous backend and a dynamic React frontend.

## 🏗 Architecture & Visual Data Flow

1. **Upload (React ➔ FastAPI):** The frontend submits multipart forms (videos/images + custom prompts) to the backend. The backend immediately yields an `HTTP 202 Accepted` alongside a UUID `job_id`, streaming the file bytes seamlessly to Gemini's Files API.
2. **Background Processing (FastAPI ➔ Gemini):** An asynchronous `BackgroundTasks` worker triggers cloud inference. The VLM executes against the strict Pydantic `VideoTrackerReport` schema, ensuring structured JSON output and isolating custom prompt responses.
3. **State Management (Worker ➔ Supabase):** Job state transitions (`pending` ➔ `processing` ➔ `completed` or `schema_error`) are exclusively written to the Supabase PostgreSQL database acting as the single source of truth, bypassing volatile local RAM cache dependencies.
4. **Polling & Rendering (React ➔ FastAPI):** The frontend polls `/api/v1/job/{job_id}`. Once marked `completed`, the React UI immediately renders the rich tracking data, event badges, and the custom VLM responses.

## ⚙️ Core Production Patterns

- **Durable DB-Backed Queue:** State lives in Supabase. A native FastAPI `@asynccontextmanager` lifespan event performs a startup recovery sweep. On container reboot, it detects orphaned jobs locked in `processing` and safely re-queues them with sentinel identifiers for graceful termination or retry.
- **Dynamic VLM Prompting:** Users can inject custom instructional prompts. The backend schema handles these robustly using an expanded Pydantic `custom_prompt_response` field, preventing Gemini from experiencing "schema blindness" while ensuring primary tracking parameters are untouched.
- **Pragmatic Budget Boundaries:** Built for free-tier resilience.
  - **Deterrence:** `slowapi` strictly limits ingress (5 requests/minute per IP) while `X-API-Key` header authentication acts as an anti-spam layer.
  - **Enforcement:** A server-side `_check_daily_budget()` guard runs prior to any Gemini call, querying Supabase for the daily UTC footprint and halting execution if the strict cap (50 jobs/day) is breached.
  - **Fault Tolerance:** Transient API errors (429s/5xxs) are automatically mitigated using `tenacity` exponential backoff retries.

## 🚀 Quick-Start Onboarding

### 1. Backend (FastAPI) Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd vision-language-pipeline/backend

# Initialize virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# Start the development server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend (React/Vite) Setup

```bash
# In a new terminal window
cd vision-language-pipeline/frontend/observability-console

# Install dependencies and start Vite
npm install
npm run dev
```

### 3. Environment Variables

Create a `.env` file in the `backend/` directory matching this exact template:

```ini
# --- AI & Cloud Integrations ---
GEMINI_API_KEY=your_google_gemini_api_key

# --- Supabase Database ---
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# --- Security ---
APP_API_KEY=your_secure_frontend_auth_key

# --- Observability (Optional) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT="vision-language-pipeline"
```
