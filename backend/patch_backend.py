import re

path = r'c:\Users\Administrator\Downloads\vision lang pipeline\vision-language-pipeline\backend\server.py'
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Fix schema
text = re.sub(
    r'summary: str = Field\(\s*description="Concise one-to-three sentence natural language summary of the video."\s*\)',
    r'summary: str = Field(\n        description="Concise one-to-three sentence natural language summary of the video."\n    )\n    custom_prompt_response: Optional[str] = Field(\n        default=None,\n        description="Detailed text answer addressing the user\'s specific custom prompt request, if one was provided."\n    )',
    text
)

# Fix instruction
text = re.sub(
    r'If no bounding box can be determined, \\n\s*"omit it. Set event_detected to true if a significant action occurs."\s*\)',
    r'If no bounding box can be determined, \n            "omit it. Set event_detected to true if a significant action occurs. "\n            "If the user provides an additional focus or question, answer it in the custom_prompt_response field."\n        )',
    text
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Backend patched.")
