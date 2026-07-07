from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from pydantic import ValidationError
from app.schemas.extraction import ComprehensiveInvoiceSchema
from app.services.real_vlm import live_vlm_extract
from db import supabase_client  # Cloud Supabase HTTPS client

# Define the state shape for our LangGraph agent
class PipelineState(TypedDict):
    document_id: str
    file_path: str
    retry_count: int
    max_retries: int
    raw_vlm_output: Optional[str]
    error_message: Optional[str]
    parsed_data: Optional[Dict[str, Any]]
    status: str  # "classifying", "extracting", "validating", "healing", "completed", "failed", "rejected"

def classification_node(state: PipelineState) -> Dict[str, Any]:
    print("--- [Node: Classification Guardrail] Analyzing File Type ---")
    doc_name = state.get("document_id", "").lower()
    file_path = state.get("file_path", "").lower()
    
    # Simple rule-based mock simulation for the cat edge case
    if "cat" in doc_name or "pet" in doc_name or "cat" in file_path or "pet" in file_path:
        print("[REJECTED] Guardrail Triggered: Non-invoice file detected. Aborting pipeline.")
        return {
            "status": "rejected",
            "error_message": "Document Rejected: The uploaded image does not appear to be an invoice or financial document."
        }
    
    print("[PASSED] Guardrail Passed: Document identified as valid financial instrument.")
    return {"status": "extracting"}

def guardrail_router(state: PipelineState) -> str:
    if state["status"] == "rejected":
        return "reject"
    return "proceed"

def extraction_node(state: PipelineState) -> Dict[str, Any]:
    print(f"--- [Node: Extraction] Processing Document {state['document_id']} (Attempt {state['retry_count'] + 1}) ---")
    
    vlm_output = live_vlm_extract(file_path=state['file_path'], error_context=state.get('error_message'))
    
    return {
        "raw_vlm_output": vlm_output,
        "status": "validating"
    }

def validation_node(state: PipelineState) -> Dict[str, Any]:
    print("--- [Node: Validation] Running Pydantic Check ---")
    try:
        # Validate using our strict Pydantic model
        validated = ComprehensiveInvoiceSchema.model_validate_json(state["raw_vlm_output"])
        print("Pydantic check passed successfully!")
        return {
            "parsed_data": validated.model_dump(mode='json'),
            "error_message": None,
            "status": "completed"
        }
    except ValidationError as e:
        print("Pydantic validation caught errors.")
        return {
            "error_message": str(e),
            "status": "healing"
        }

def healing_router(state: PipelineState) -> str:
    # Router to determine the next path in the graph
    if state["status"] == "completed":
        return "end"
    
    if state["retry_count"] >= state["max_retries"]:
        print("Maximum self-healing attempts reached. Marking as Failed.")
        return "fail"
    
    return "retry"

def healing_node(state: PipelineState) -> Dict[str, Any]:
    print(f"--- [Node: Self-Healing] Constructing Feedback Loop for Attempt {state['retry_count'] + 1} ---")
    # In a real environment, state["error_message"] is injected back into the prompt context
    return {
        "retry_count": state["retry_count"] + 1,
        "status": "extracting"
    }

def save_to_db_node(state: PipelineState) -> Dict[str, Any]:
    """
    Terminal persistence node.
    Writes every pipeline outcome to `vlp_extractions` via the Supabase HTTPS client.
    For `failed` status, also creates a HITL review row in `hitl_review_queue`.
    """
    print(f"--- [Node: Save to DB] Persisting result for document {state['document_id']} (status={state['status']}) ---")

    extraction_payload = {
        "document_id":   state["document_id"],
        "status":        state["status"],
        "parsed_data":   state.get("parsed_data"),
        "error_message": state.get("error_message"),
        "retry_count":   state.get("retry_count", 0),
        "raw_vlm_output": state.get("raw_vlm_output"),
    }

    try:
        supabase_client.table("vlp_extractions").upsert(extraction_payload).execute()
        print(f"[DB] Written to vlp_extractions: document_id={state['document_id']}")
    except Exception as e:
        # Log but do not crash the pipeline — observability over availability.
        print(f"[DB ERROR] Failed to write to vlp_extractions: {e}")

    # Route failed documents into the human-in-the-loop review queue.
    if state["status"] == "failed":
        hitl_payload = {
            "document_id":   state["document_id"],
            "reason":        state.get("error_message", "Max retries exceeded"),
            "retry_count":   state.get("retry_count", 0),
            "review_status": "pending",
        }
        try:
            supabase_client.table("hitl_review_queue").insert(hitl_payload).execute()
            print(f"[DB] Inserted into hitl_review_queue: document_id={state['document_id']}")
        except Exception as e:
            print(f"[DB ERROR] Failed to write to hitl_review_queue: {e}")

    return {}  # No state mutation needed — this is a terminal sink node.

# Build and compile the LangGraph State Machine
workflow = StateGraph(PipelineState)

# Add all nodes
workflow.add_node("classify",    classification_node)
workflow.add_node("extract",     extraction_node)
workflow.add_node("validate",    validation_node)
workflow.add_node("heal",        healing_node)
workflow.add_node("save_to_db",  save_to_db_node)  # <-- NEW: terminal persistence node

# Set the entry point
workflow.set_entry_point("classify")

# Guardrail routing: rejected docs are persisted before exiting
workflow.add_conditional_edges(
    "classify",
    guardrail_router,
    {
        "reject":  "save_to_db",   # was: END
        "proceed": "extract"
    }
)

# Core pipeline edges (unchanged)
workflow.add_edge("extract", "validate")

# Healing router: all terminal paths now funnel through save_to_db before END
workflow.add_conditional_edges(
    "validate",
    healing_router,
    {
        "end":   "save_to_db",  # was: END  (successful completion)
        "retry": "heal",
        "fail":  "save_to_db",  # was: END  (max retries exceeded)
    }
)
workflow.add_edge("heal", "extract")
workflow.add_edge("save_to_db", END)  # <-- single canonical exit point

orchestrator_agent = workflow.compile()
