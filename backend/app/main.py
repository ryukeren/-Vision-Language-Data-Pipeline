import os
from dotenv import load_dotenv
load_dotenv()  # Inject .env into os.environ for LangSmith tracing

print(f"LangSmith Tracing Enabled: {os.environ.get('LANGSMITH_TRACING')}")
print(f"LangSmith Endpoint: {os.environ.get('LANGSMITH_ENDPOINT')}")
import uuid
import tempfile
import traceback

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.services.orchestrator import orchestrator_agent

app = FastAPI(title="Vision-Language Pipeline Gateway")

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for observability (replace with Supabase in full production)
JOB_STORE = {}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def run_agentic_pipeline(doc_id: str, file_path: str):
    """
    Runs the LangGraph agentic pipeline in a background thread.
    Catches any unhandled exceptions to prevent silent hangs.
    """
    JOB_STORE[doc_id] = {"status": "extracting", "logs": ["Started extraction..."]}

    initial_state = {
        "document_id": doc_id,
        "file_path": file_path,
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
        # Bug Fix #3: Catch silent crashes and surface them to the status endpoint
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

@app.post("/api/v1/upload")
async def upload_and_process(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Bug Fix #1 & #2: Accept real file bytes from the frontend.
    Save them to a Docker-safe temporary file, then pass the real path to the pipeline.
    """
    generated_doc_id = str(uuid.uuid4())
    JOB_STORE[generated_doc_id] = {"status": "queued"}

    # Read the uploaded file bytes
    file_bytes = await file.read()

    # Determine suffix from original filename for correct MIME detection
    original_name = file.filename or "upload.png"
    suffix = os.path.splitext(original_name)[-1] or ".png"

    # Save to a real temp file that will exist inside the Docker container filesystem
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()

    print(f"[Upload] Saved uploaded file '{original_name}' to temp path: {tmp.name}")

    background_tasks.add_task(run_agentic_pipeline, generated_doc_id, tmp.name)

    return {
        "message": "Pipeline initialized.",
        "document_id": generated_doc_id,
        "status": "queued"
    }

# Keep old JSON endpoint for backwards compatibility (local testing with filename strings)
@app.post("/api/v1/process")
async def process_document_by_path(background_tasks: BackgroundTasks, payload: dict):
    """
    Legacy endpoint: accepts a JSON body with a 'filename' key.
    Only works when the file physically exists on the backend filesystem (local dev only).
    """
    from pydantic import BaseModel
    filename = payload.get("filename", "")
    generated_doc_id = str(uuid.uuid4())
    JOB_STORE[generated_doc_id] = {"status": "queued"}
    background_tasks.add_task(run_agentic_pipeline, generated_doc_id, filename)
    return {
        "message": "Pipeline initialized.",
        "document_id": generated_doc_id,
        "status": "queued"
    }

@app.get("/api/v1/status/{document_id}")
def get_status(document_id: str):
    if document_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Document ID not found.")
    return JOB_STORE[document_id]
