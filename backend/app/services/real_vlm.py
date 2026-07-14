import mimetypes
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.schemas.extraction import ComprehensiveInvoiceSchema
from config import settings  # reads GEMINI_API_KEY from .env via Pydantic Settings

# Retry policy: intercept transient Gemini API errors only.
# Exponential backoff: 2s → 4s → 8s → 15s (cap), up to 5 attempts total.
_RETRY_POLICY = retry(
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)


@_RETRY_POLICY
def live_vlm_extract(file_path: str, error_context: str = None, user_prompt: str = None) -> str:
    """
    Sends an actual local image file to Gemini 2.5 Flash, 
    enforcing our strict Pydantic schema structure.

    Args:
        file_path:    Path to the local image file to analyze.
        error_context: On a self-healing retry, the previous Pydantic error string
                       is injected here so Gemini can correct its output.
        user_prompt:  Optional custom extraction directive from the user. Falls back
                      to a standard comprehensive extraction instruction if not provided.
    """
    # Read API key from Pydantic Settings (already loaded from .env at startup)
    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env file.")

    client = genai.Client(api_key=api_key)
    
    # Read the local file bytes
    with open(file_path, "rb") as f:
        image_bytes = f.read()
        
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "image/png"

    # Define base prompt instructions
    system_instruction = (
        "You are an expert data extraction agent. Analyze the provided invoice image "
        "and extract all structured data matching the requested JSON schema accurately."
    )
    
    # Use the custom user prompt if provided; otherwise fall back to a sensible default.
    # This is the core of the custom prompt feature — the user's instruction becomes
    # the primary directive that guides Gemini's extraction focus.
    base_prompt = user_prompt.strip() if user_prompt and user_prompt.strip() \
        else "Extract all invoice details, line items, billing information, and tax calculations precisely."
    
    # On a LangGraph self-healing retry, append the exact Pydantic error context
    # so Gemini knows what to fix in its next attempt.
    if error_context:
        final_prompt = f"{base_prompt}\n\nWARNING: Your previous attempt failed schema validation with this error:\n{error_context}\nFix your output to resolve this error."
    else:
        final_prompt = base_prompt

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                final_prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                # Force Gemini to return data matching our exact Pydantic structural contract
                response_mime_type="application/json",
                response_schema=ComprehensiveInvoiceSchema,
                temperature=0.1
            ),
        )
        return response.text
    except Exception as e:
        print(f"API Error during extraction: {e}")
        raise e
