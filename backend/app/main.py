import os
from dotenv import load_dotenv
load_dotenv()  # Inject .env into os.environ for LangSmith tracing

print(f"LangSmith Tracing Enabled: {os.environ.get('LANGSMITH_TRACING')}")
print(f"LangSmith Endpoint: {os.environ.get('LANGSMITH_ENDPOINT')}")
import uuid
import tempfile
import traceback

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from typing import Optional
from app.services.orchestrator import orchestrator_agent
from db import supabase_client

# ─────────────────────────────────────────────────────────────────────────────
# Invoice Pipeline Router
# Mounted onto the main server.py FastAPI app via app.include_router()
# ─────────────────────────────────────────────────────────────────────────────

invoice_router = APIRouter(tags=["Invoice Pipeline"])

# In-memory store for observability (keyed by document_id)
JOB_STORE = {}


@invoice_router.get("/invoice/health")
def invoice_health_check():
    return {"status": "healthy", "pipeline": "invoice"}


def run_agentic_pipeline(doc_id: str, file_path: str, user_prompt: Optional[str] = None):
    """
    Runs the LangGraph agentic pipeline in a background thread.
    Catches any unhandled exceptions to prevent silent hangs.
    """
    JOB_STORE[doc_id] = {"status": "extracting", "logs": ["Started extraction..."]}

    initial_state = {
        "document_id": doc_id,
        "file_path": file_path,
        "user_prompt": user_prompt or None,
        "retry_count": 0,
        "max_retries": 2,
        "raw_vlm_output": None,
        "error_message": None,
        "parsed_data": None,
        "status": "classifying"
    }
    try:
        final_output = orchestrator_agent.invoke(initial_state)

        JOB_STORE[doc_id] = {
            "status": final_output["status"],
            "parsed_data": final_output.get("parsed_data"),
            "error_message": final_output.get("error_message"),
            "retries": final_output.get("retry_count", 0)
        }

    except Exception as e:
        # Catch silent crashes and surface them to the status endpoint
        error_detail = traceback.format_exc()
        print(f"[PIPELINE CRASH] doc_id={doc_id}\n{error_detail}")
        JOB_STORE[doc_id] = {
            "status": "failed",
            "parsed_data": None,
            "error_message": f"Internal pipeline error: {str(e)}",
            "retries": initial_state["retry_count"]
        }
    finally:
        # Clean up the temporary file after processing, regardless of success or failure
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[Cleanup] Removed temp file: {file_path}")
            except Exception:
                pass


@invoice_router.post("/api/v1/upload")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None, description="Custom extraction directive for Gemini. Defaults to a comprehensive invoice extraction instruction if blank."),
):
    """
    Accepts real file bytes from the frontend via multipart/form-data.
    Saves them to a temporary file, then launches the LangGraph pipeline
    as a background task with the optional custom prompt.
    """
    generated_doc_id = str(uuid.uuid4())
    JOB_STORE[generated_doc_id] = {"status": "queued"}

    # Read the uploaded file bytes
    file_bytes = await file.read()

    # Determine suffix from original filename for correct MIME detection
    original_name = file.filename or "upload.png"
    suffix = os.path.splitext(original_name)[-1] or ".png"

    # Save to a real temp file that will exist inside the container filesystem
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()

    print(f"[Upload] Saved '{original_name}' to temp path: {tmp.name}")
    if prompt:
        print(f"[Upload] Custom prompt received: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    background_tasks.add_task(run_agentic_pipeline, generated_doc_id, tmp.name, prompt)

    return {
        "message": "Pipeline initialized.",
        "document_id": generated_doc_id,
        "status": "queued"
    }


# Keep old JSON endpoint for backwards compatibility (local testing with filename strings)
@invoice_router.post("/api/v1/process")
async def process_document_by_path(background_tasks: BackgroundTasks, payload: dict):
    """
    Legacy endpoint: accepts a JSON body with a 'filename' key.
    Only works when the file physically exists on the backend filesystem (local dev only).
    """
    filename = payload.get("filename", "")
    generated_doc_id = str(uuid.uuid4())
    JOB_STORE[generated_doc_id] = {"status": "queued"}
    background_tasks.add_task(run_agentic_pipeline, generated_doc_id, filename)
    return {
        "message": "Pipeline initialized.",
        "document_id": generated_doc_id,
        "status": "queued"
    }


@invoice_router.get("/api/v1/status/{document_id}")
def get_status(document_id: str):
    if document_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Document ID not found.")
    return JOB_STORE[document_id]
