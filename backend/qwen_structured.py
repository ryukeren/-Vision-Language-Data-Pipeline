"""
Phase 2: Qwen2.5-VL Structured Output Layer
============================================
Adds local Pydantic schema enforcement to the Phase 1 inference engine.

Since we are now running Qwen2.5-VL locally (not on Google's cloud), we cannot
rely on `response_mime_type="application/json"` to force structured output.
Instead, we:
  1. Inject the exact target JSON schema directly into the prompt.
  2. Post-process the raw model output to strip accidental markdown wrappers.
  3. Validate the cleaned JSON against our Pydantic model.
  4. Return a typed `VideoTrackerReport` object — or a clean fallback dict on failure.

Prerequisites (same as Phase 1):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate qwen-vl-utils decord pydantic
"""

import re
import json
from typing import Optional
import torch
from pydantic import BaseModel, ValidationError, Field
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Pydantic Schema Definition
# ─────────────────────────────────────────────────────────────────────────────

class TrackedObject(BaseModel):
    """
    Represents a single object detected and tracked in the video.
    """
    label: str = Field(
        description="The class label of the detected object (e.g., 'person', 'car', 'dog')."
    )
    timestamp_seconds: Optional[float] = Field(
        default=None,
        description="The approximate video timestamp (in seconds) when this object was first observed."
    )
    box_2d: Optional[list[float]] = Field(
        default=None,
        description=(
            "Normalized bounding box coordinates as [ymin, xmin, ymax, xmax]. "
            "All values must be floats between 0.0 and 1.0."
        )
    )


class VideoTrackerReport(BaseModel):
    """
    The top-level structured report for a single video analysis run.
    """
    event_detected: bool = Field(
        description="True if any significant event or activity was detected in the video."
    )
    summary: str = Field(
        description="A concise, one-to-three sentence natural language summary of the entire video content."
    )
    tracked_objects: list[TrackedObject] = Field(
        default_factory=list,
        description="A list of all distinct objects identified and tracked throughout the video."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Schema-Injected Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_schema_prompt() -> str:
    """
    Generates the schema-injection prompt string.

    This is the local equivalent of Google's `response_mime_type="application/json"`.
    We explicitly embed the JSON structure we expect the model to produce, so the
    model has a precise contract to fulfill.
    """
    schema_str = json.dumps(VideoTrackerReport.model_json_schema(), indent=2)

    return f"""Analyze this video and respond ONLY with a single, valid JSON object.
Do NOT include any explanatory text, markdown code fences, or ```json wrappers.
Your entire response must be parseable raw JSON conforming EXACTLY to this schema:

{schema_str}

IMPORTANT RULES:
- Output only the raw JSON object. Nothing else before or after it.
- `box_2d` values must be normalized floats between 0.0 and 1.0, in order [ymin, xmin, ymax, xmax].
- If a bounding box or timestamp is not available, use null for that field.
- `tracked_objects` must be a list (use an empty list [] if no objects are detected).
- `event_detected` must be a boolean true or false (not a string).
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Output Cleaning & Validation
# ─────────────────────────────────────────────────────────────────────────────

def _clean_raw_output(raw_text: str) -> str:
    """
    Cleans accidental markdown wrappers that LLMs often produce even when instructed not to.

    Handles patterns like:
        ```json { ... } ```
        ``` { ... } ```
        Leading/trailing whitespace
    """
    # Strip leading/trailing whitespace first
    cleaned = raw_text.strip()

    # Pattern 1: Remove ```json ... ``` or ``` ... ``` fences
    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
    match = fence_pattern.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Pattern 2: If a JSON object starts somewhere mid-text (model hallucinated preamble),
    # find the first '{' and extract from there.
    if not cleaned.startswith("{"):
        brace_index = cleaned.find("{")
        if brace_index != -1:
            cleaned = cleaned[brace_index:]

    # Pattern 3: Trim any trailing content after the final closing brace
    last_brace = cleaned.rfind("}")
    if last_brace != -1:
        cleaned = cleaned[:last_brace + 1]

    return cleaned


def _validate_output(raw_text: str) -> VideoTrackerReport | dict:
    """
    Cleans raw model output and validates it against the VideoTrackerReport Pydantic schema.

    Returns:
        VideoTrackerReport: A validated, typed object on success.
        dict: A clean fallback error dictionary if validation fails.
    """
    print("\n🔍 Post-processing raw model output...")
    cleaned = _clean_raw_output(raw_text)

    try:
        report = VideoTrackerReport.model_validate_json(cleaned)
        print("✅ Schema validation passed.")
        return report
    except ValidationError as e:
        print("⚠️  Pydantic validation FAILED. Raw output printed below for debugging:")
        print("-" * 40)
        print(f"RAW OUTPUT:\n{raw_text}")
        print(f"\nCLEANED ATTEMPT:\n{cleaned}")
        print(f"\nVALIDATION ERRORS:\n{e}")
        print("-" * 40)
        return {
            "error": "schema_validation_failed",
            "pydantic_errors": e.errors(),
            "raw_model_output": raw_text,
        }
    except json.JSONDecodeError as e:
        print("⚠️  JSON parsing FAILED. The model output is not valid JSON.")
        print("-" * 40)
        print(f"RAW OUTPUT:\n{raw_text}")
        print(f"JSON ERROR: {e}")
        print("-" * 40)
        return {
            "error": "json_decode_failed",
            "json_error": str(e),
            "raw_model_output": raw_text,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Model Initialization (unchanged from Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_model():
    """
    Initializes the Qwen2.5-VL-7B-Instruct model and processor locally.
    Uses bfloat16 and device_map="auto" for optimal VRAM usage.
    """
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading model '{model_id}' into memory... (This may take several minutes)")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)
    print("✅ Model and processor loaded successfully!")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Core Structured Inference Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video_local(
    model,
    processor,
    video_path: str,
    prompt: str = None,
) -> VideoTrackerReport | dict:
    """
    Analyzes a local video file and returns a validated, structured VideoTrackerReport.

    The schema-injection prompt is built automatically. An optional `prompt`
    argument can be used to provide additional analytical context or focus
    (e.g., "Focus specifically on any vehicles present").

    Args:
        model:       The loaded Qwen2.5-VL model instance.
        processor:   The loaded AutoProcessor instance.
        video_path:  Absolute or relative path to a local .mp4 file.
        prompt:      Optional additional instruction to append to the base schema prompt.

    Returns:
        VideoTrackerReport: A fully validated Pydantic object on success.
        dict:               A fallback error dictionary on schema/JSON failure.
    """
    print(f"\n🎥 Analyzing video: {video_path}")

    # Build the schema-injected prompt. Append any extra user context if provided.
    base_prompt = _build_schema_prompt()
    full_prompt = f"{base_prompt}\n\nAdditional context: {prompt}" if prompt else base_prompt

    # 1. Structure the message payload with VRAM safety caps
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 100352,   # Caps resolution per frame to prevent OOM
                    "fps": 1.0,             # Sample 1 frame per second
                },
                {
                    "type": "text",
                    "text": full_prompt,
                },
            ],
        }
    ]

    print("⏳ Formatting inputs and extracting video frames...")

    # 2. Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 3. Process visual inputs via qwen-vl-utils
    image_inputs, video_inputs = process_vision_info(messages)

    # 4. Tokenize and move to device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print("🧠 Running local inference...")

    # 5. Generate — allow enough tokens for a full JSON response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,   # Sufficient headroom for detailed JSON output
            do_sample=False,      # Greedy decoding: deterministic, consistent JSON
            temperature=None,     # Must be None when do_sample=False
            top_p=None,           # Must be None when do_sample=False
        )

    # 6. Trim input tokens from output, then decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # 7. Validate and return a typed VideoTrackerReport (or fallback dict on failure)
    result = _validate_output(raw_output)

    print("\n" + "=" * 50)
    if isinstance(result, VideoTrackerReport):
        print("📊 STRUCTURED REPORT:")
        print("=" * 50)
        print(result.model_dump_json(indent=2))
    else:
        print("🚨 FALLBACK ERROR REPORT:")
        print("=" * 50)
        print(json.dumps(result, indent=2))
    print("=" * 50 + "\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Initialize the AI once (expensive — do not re-initialize per request)
    local_model, local_processor = initialize_model()

    # Step 2: Run a structured analysis
    # Replace "sample_video.mp4" with a real file path on your system.
    test_video_path = "sample_video.mp4"

    # Optional: provide extra analytical focus instructions.
    # Leave as None to use the base schema prompt only.
    extra_context = "Focus on detecting any human activity and vehicles."

    result = analyze_video_local(
        model=local_model,
        processor=local_processor,
        video_path=test_video_path,
        prompt=extra_context,
    )

    # The result is now a fully typed Python object you can use downstream
    if isinstance(result, VideoTrackerReport):
        print(f"Event detected: {result.event_detected}")
        print(f"Objects tracked: {len(result.tracked_objects)}")
