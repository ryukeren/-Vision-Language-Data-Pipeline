"""
Phase 1: Qwen2.5-VL Core Inference Engine (Standalone)

Installation Requirements:
Run the following pip command before executing this script:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate qwen-vl-utils decord torchvision

Note: `decord` and `torchvision` are required by `qwen-vl-utils` to process video files.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def initialize_model():
    """
    Initializes the Qwen2.5-VL-7B-Instruct model and processor locally using PyTorch.
    Uses device_map="auto" to automatically distribute weights across available GPUs,
    and bfloat16 to optimize memory usage.
    """
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading model '{model_id}' into memory... (This may take several minutes)")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("✅ Model and processor loaded successfully!")
    return model, processor

def analyze_video_local(model, processor, video_path: str, prompt: str):
    """
    Analyzes a local video file using the initialized Qwen2.5-VL model.
    Caps memory usage by restricting max_pixels and fps.
    """
    print(f"\n🎥 Analyzing video: {video_path}")
    print(f"💬 Prompt: {prompt}")

    # 1. Structure the message payload
    # By specifying max_pixels and fps, we strictly bound the VRAM usage for the video tokens.
    # Without these limits, a long video will cause an Out of Memory (OOM) error.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 100352, # Caps resolution per frame to prevent OOM
                    "fps": 1.0,           # Samples 1 frame per second of video
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]

    print("⏳ Formatting inputs and extracting frames...")
    # 2. Prepare the prompt text
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 3. Process the video file using qwen-vl-utils
    image_inputs, video_inputs = process_vision_info(messages)

    # 4. Tokenize and move to device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move all input tensors to the same device as the model
    inputs = inputs.to(model.device)

    print("🧠 Generating response (Inference)...")
    # 5. Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    # 6. Decode output
    # We trim the input tokens from the generated_ids to only print the new text
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n" + "="*50)
    print("📝 OUTPUT:")
    print("="*50)
    print(output_text[0])
    print("="*50 + "\n")

if __name__ == "__main__":
    # Example usage:
    # 1. Initialize the AI
    local_model, local_processor = initialize_model()
    
    # 2. Provide a sample video and prompt
    # Replace 'sample_video.mp4' with an actual file path on your system.
    test_video_path = "sample_video.mp4" 
    test_prompt = "Describe the events taking place in this video."
    
    try:
        analyze_video_local(local_model, local_processor, test_video_path, test_prompt)
    except Exception as e:
        print(f"❌ Failed to analyze video: {e}")
