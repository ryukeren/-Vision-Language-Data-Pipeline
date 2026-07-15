"""
Phase 3 (Revised): Hybrid Cloud-Edge Video Analytics Server
=============================================================
Architecture pivot: CPU-only laptop → cloud inference via Gemini 2.5 Flash.

What changed from the original server.py:
  ✗ REMOVED: torch, transformers, qwen-vl-utils, local model loading
  ✓ KEPT:    FastAPI structure, port 8001, multipart UploadFile streaming,
             shutil temp-file handling, finally-block cleanup,
             local Pydantic VideoTrackerReport validation

How it works:
  1. Upload arrives as multipart form-data (.mp4 / image / etc.)
  2. File is streamed to a local temp file (qwen-vl-utils no longer needed,
     but Gemini SDK also requires a path OR bytes — we use bytes here).
  3. File bytes are uploaded to the Gemini Files API (handles files > 20 MB).
  4. Gemini 2.5 Flash processes the video with response_schema enforcement
     so the cloud returns pre-validated JSON — no regex cleaning needed.
  5. We parse the response through VideoTrackerReport.model_validate_json()
     as a local double-check layer.
  6. Temp file is deleted in a finally block; the Gemini uploaded file is also
     deleted from cloud storage to avoid accumulating usage quota.

Run command:
    uvicorn server:app --host 0.0.0.0 --port 8001 --reload

Dependencies (already in requirements.txt):
    fastapi, uvicorn[standard], google-genai, google-api-core,
    pydantic, pydantic-settings, python-multipart
"""

import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings  # reads GEMINI_API_KEY from .env via Pydantic Settings
from db import supabase_client
from app.main import invoice_router  # Invoice pipeline router


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Pydantic Schema (VideoTrackerReport)
#
# Kept identical to qwen_structured.py so both servers speak the same contract.
# The schema is passed BOTH to Gemini (cloud-side enforcement) and to
# model_validate_json() below (local double-check).
# ─────────────────────────────────────────────────────────────────────────────

class TrackedObject(BaseModel):
    label: str = Field(
        description="Class label of the detected object (e.g., 'person', 'car', 'dog')."
    )
    timestamp_seconds: Optional[float] = Field(
        default=None,
        description="Approximate video timestamp (seconds) when the object was first observed."
    )
    box_2d: Optional[list[float]] = Field(
        default=None,
        description=(
            "Normalized bounding box [ymin, xmin, ymax, xmax]. "
            "All values must be floats between 0.0 and 1.0."
        )
    )


class VideoTrackerReport(BaseModel):
    event_detected: bool = Field(
        description="True if any significant event or activity was detected."
    )
    summary: str = Field(
        description="Concise one-to-three sentence natural language summary of the video."
    )
    tracked_objects: list[TrackedObject] = Field(
        default_factory=list,
        description="All distinct objects identified and tracked throughout the video."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Global Gemini Client (lightweight — no GPU, no model weights)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_CLIENT: Optional[genai.Client] = None

# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Job Store
# Holds the status and result of every video analysis job during the server's
# lifetime. Keyed by job_id (UUID string).
# Structure per job:
#   { "status": "pending" | "processing" | "completed" | "failed",
#     "result": <VideoTrackerReport dict> | None,
#     "error":  <str> | None }
# ─────────────────────────────────────────────────────────────────────────────

jobs_db: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2b: Security — Rate Limiter + API Key Auth
#
# Layer 1 — SlowAPI rate limiter (5 requests / minute per client IP).
#   Prevents spam bots from hammering the Gemini API endpoint.
#   The limiter key is the client's IP address (get_remote_address).
#
# Layer 2 — X-API-Key header authentication.
#   Only clients that present the correct APP_API_KEY secret are allowed
#   to submit video jobs. The polling route is intentionally left open.
# ─────────────────────────────────────────────────────────────────────────────

# Rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address, default_limits=[])

# Header scheme: reads the raw value of the 'X-API-Key' header
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency — raises 401 if the X-API-Key header is missing
    or does not match the APP_API_KEY environment variable.
    Apply with: dependencies=[Depends(verify_api_key)]
    """
    expected = settings.app_api_key
    if not expected:
        # No key configured on the server → auth is disabled (dev mode)
        return "dev-mode-no-key-set"
    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide it via the X-API-Key header.",
        )
    return api_key


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Lifespan Manager (startup / shutdown)
#
# Replaces the heavy model-loading lifespan from the original server.py.
# All we need to initialise now is a single lightweight API client object.
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GEMINI_CLIENT

    print("=" * 58)
    print("[STARTUP] Hybrid Cloud-Edge Video Analytics Server")
    print("=" * 58)

    if not settings.gemini_api_key:
        # Fail fast — no point starting the server without credentials
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Please add it to backend/.env and restart the server."
        )

    GEMINI_CLIENT = genai.Client(api_key=settings.gemini_api_key)
    print("[OK] Gemini API client initialised (no local GPU required).")
    print("[OK] Model: gemini-2.5-flash  |  Invoice + Video pipelines unified")
    print("=" * 58)

    yield  # Server is live here

    # Shutdown cleanup
    GEMINI_CLIENT = None
    print("[SHUTDOWN] Server stopped. Gemini client released.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🎥 Hybrid Cloud-Edge Video Analytics API",
    description=(
        "A CPU-friendly video analysis API. Uploads are streamed locally, "
        "then sent to Gemini 2.5 Flash for cloud inference. "
        "Responses are validated locally against VideoTrackerReport before being returned."
    ),
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire the SlowAPI limiter into the FastAPI app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4b: Mount Invoice Pipeline Router
# All invoice endpoints (/api/v1/upload, /api/v1/status, etc.) are served here.
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(invoice_router)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Lightweight heartbeat — confirms the Gemini client is initialised."""
    is_ready = GEMINI_CLIENT is not None
    return {
        "status": "healthy" if is_ready else "initialising",
        "architecture": "hybrid-cloud-edge",
        "inference_backend": "gemini-2.5-flash (cloud)",
        "local_gpu_required": False,
        "client_ready": is_ready,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Cloud Inference Helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_gemini_video_analysis(tmp_path: str, prompt: Optional[str]) -> VideoTrackerReport:
    """
    Uploads the video file to Gemini's Files API, runs inference with
    structured JSON enforcement, then validates the response locally.

    Uses Gemini's Files API (rather than inline bytes) because:
      - Videos > 20 MB must be uploaded via the Files API.
      - The Files API returns a stable URI that Gemini can stream directly,
        avoiding holding the entire file in memory during the API call.

    Args:
        tmp_path: Absolute path to the temporary video file on disk.
        prompt:   Optional analytical focus instruction from the user.

    Returns:
        VideoTrackerReport: A locally-validated Pydantic object.

    Raises:
        HTTPException 422: If Gemini returns malformed JSON.
        HTTPException 500: If the Gemini API call itself fails.
    """
    uploaded_file = None

    try:
        # Step 1 — Upload the video to Gemini Files API
        print(f"[UPLOAD] Uploading '{tmp_path}' to Gemini Files API...")
        with open(tmp_path, "rb") as video_file:
            uploaded_file = GEMINI_CLIENT.files.upload(
                file=video_file,
                config=types.UploadFileConfig(
                    mime_type="video/mp4",
                    display_name=os.path.basename(tmp_path),
                )
            )
        print(f"[UPLOAD] Complete. File URI: {uploaded_file.uri}")

        # Step 2 — Poll until the file is fully processed by Gemini
        # (Large files may take a moment to become ACTIVE)
        print("[POLL] Waiting for Gemini to process the video file...")
        import time
        max_wait_seconds = 60
        poll_interval = 3
        waited = 0
        while waited < max_wait_seconds:
            file_state = GEMINI_CLIENT.files.get(name=uploaded_file.name)
            if file_state.state.name == "ACTIVE":
                print("[POLL] File is ACTIVE and ready for inference.")
                break
            if file_state.state.name == "FAILED":
                raise HTTPException(
                    status_code=500,
                    detail="Gemini Files API failed to process the uploaded video."
                )
            time.sleep(poll_interval)
            waited += poll_interval
        else:
            raise HTTPException(
                status_code=500,
                detail="Timed out waiting for Gemini to process the video file."
            )

        # Step 3 — Build the prompt
        base_instruction = (
            "You are an expert video analytics agent. "
            "Analyze the provided video and extract all observable events and objects. "
            "Respond ONLY with a structured JSON object matching the provided schema exactly."
        )
        user_prompt = "Analyze this video and extract all events and tracked objects."
        if prompt:
            user_prompt += f"\n\nAdditional focus: {prompt}"

        # Step 4 — Call Gemini with cloud-side schema enforcement
        # response_mime_type="application/json" + response_schema=VideoTrackerReport
        # instructs Gemini to constrain its token sampling to valid JSON matching
        # our Pydantic schema — the cloud does the JSON heavy lifting for us.
        print("[INFERENCE] Sending to Gemini 2.5 Flash...")
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type="video/mp4",
                ),
                user_prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=base_instruction,
                response_mime_type="application/json",
                response_schema=VideoTrackerReport,
                temperature=0.1,
            ),
        )

        raw_json: str = response.text
        print(f"[INFERENCE] Response received ({len(raw_json)} chars).")

        # Step 5 — Local double-check validation
        # Even though Gemini enforces the schema cloud-side, we validate locally
        # to catch any edge-case drift and get a fully-typed Python object.
        try:
            report = VideoTrackerReport.model_validate_json(raw_json)
            print("[VALID] Local Pydantic validation passed.")
            return report
        except ValidationError as val_err:
            print(f"[WARN] Local validation failed. Raw output:\n{raw_json}")
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Gemini returned JSON that failed local schema validation.",
                    "pydantic_errors": val_err.errors(),
                    "raw_gemini_output": raw_json,
                },
            ) from val_err

    except APIError as api_err:
        print(f"[ERROR] Gemini API error: {api_err}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error: {str(api_err)}",
        ) from api_err

    finally:
        # Always delete the uploaded file from Gemini cloud storage
        # to avoid accumulating quota usage between requests.
        if uploaded_file is not None:
            try:
                GEMINI_CLIENT.files.delete(name=uploaded_file.name)
                print(f"[CLEANUP] Deleted Gemini cloud file: {uploaded_file.name}")
            except Exception as cleanup_err:
                print(f"[WARN] Could not delete Gemini cloud file: {cleanup_err}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Background Worker
# Runs the full Gemini pipeline in a background thread so the HTTP response
# is returned immediately (202 Accepted) without blocking the server.
# ─────────────────────────────────────────────────────────────────────────────

def process_video_task(job_id: str, tmp_path: str, prompt: Optional[str]):
    """
    Background worker — runs in a FastAPI background thread.
    Calls Gemini, validates the result, persists to Supabase, then
    updates the in-memory jobs_db entry so the polling endpoint can
    return the final data.
    """
    jobs_db[job_id]["status"] = "processing"
    print(f"[BG:{job_id}] Worker started.")

    try:
        report = _run_gemini_video_analysis(tmp_path=tmp_path, prompt=prompt)

        # Persist to Supabase
        if supabase_client:
            try:
                supabase_client.table("vlp_extractions").insert({
                    "document_id": job_id,
                    "status": "completed",
                    "parsed_data": report.model_dump(mode="json"),
                    "job_type": "video",
                    "prompt_used": prompt or "default",
                }).execute()
            except Exception as db_err:
                print(f"[BG:{job_id}] DB write failed (non-fatal): {db_err}")

        jobs_db[job_id] = {
            "status": "completed",
            "result": report.model_dump(mode="json"),
            "error": None,
        }
        print(f"[BG:{job_id}] Completed successfully.")

    except HTTPException as http_exc:
        jobs_db[job_id] = {
            "status": "failed",
            "result": None,
            "error": str(http_exc.detail),
        }
        print(f"[BG:{job_id}] Failed with HTTPException: {http_exc.detail}")

    except Exception as exc:
        jobs_db[job_id] = {
            "status": "failed",
            "result": None,
            "error": f"Unexpected error: {str(exc)}",
        }
        print(f"[BG:{job_id}] Failed with unexpected error: {exc}")

    finally:
        # Guaranteed cleanup of the local temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"[BG:{job_id}] Deleted local temp file: '{tmp_path}'")
            except OSError as err:
                print(f"[BG:{job_id}] Could not delete temp file: {err}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Core Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/analyze-video",
    tags=["Video Analytics"],
    status_code=202,
    summary="Submit a video for async analysis — returns a job_id immediately",
    dependencies=[Depends(verify_api_key)],   # Layer 2: API key auth
    responses={
        202: {"description": "Job accepted and queued for background processing"},
        401: {"description": "Missing or invalid X-API-Key header"},
        415: {"description": "Unsupported file type"},
        429: {"description": "Rate limit exceeded (5 requests/minute per IP)"},
        503: {"description": "Gemini client not yet initialised"},
    },
)
@limiter.limit("5/minute")   # Layer 1: rate limiting
async def analyze_video(
    request: Request,           # required by SlowAPI to resolve the client IP
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The video file to analyze (.mp4 recommended)."),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional focus instruction. E.g., 'Focus on detecting vehicles.'",
    ),
):
    """
    ### POST /api/v1/analyze-video  →  202 Accepted

    Immediately returns a `job_id`. The Gemini pipeline runs in the background.
    Poll `GET /api/v1/job/{job_id}` to check status and retrieve results.
    """
    # Guard: client must be ready
    if GEMINI_CLIENT is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini client is still initialising. Please retry in a moment.",
        )

    # Validate file extension
    original_filename = file.filename or "upload.mp4"
    file_extension = os.path.splitext(original_filename)[-1].lower()
    SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file_extension}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # Stream upload to a temp file (chunk-by-chunk, no RAM spike)
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension,
        prefix="vlp_video_",
    ) as tmp_file:
        tmp_path = tmp_file.name
        shutil.copyfileobj(file.file, tmp_file)

    print(f"\n[REQUEST] Received: '{original_filename}' -> temp: '{tmp_path}'")

    # Create a job entry in the in-memory store
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {"status": "pending", "result": None, "error": None}

    # Enqueue the background worker — returns immediately to the client
    background_tasks.add_task(process_video_task, job_id, tmp_path, prompt)

    print(f"[REQUEST] Queued background job: {job_id}")
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "pending"},
    )


@app.get(
    "/api/v1/job/{job_id}",
    tags=["Video Analytics"],
    summary="Poll a video analysis job for its current status and result",
    responses={
        200: {"description": "Job status returned (pending / processing / completed / failed)"},
        404: {"description": "No job with this ID exists"},
    },
)
def get_job_status(job_id: str):
    """
    ### GET /api/v1/job/{job_id}

    Returns the current state of the background job:
    - `pending`    — queued but not started
    - `processing` — Gemini API call in flight
    - `completed`  — `result` contains the full VideoTrackerReport
    - `failed`     — `error` contains the failure reason
    """
    job = jobs_db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
