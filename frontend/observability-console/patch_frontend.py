import re

path = r'c:\Users\Administrator\Downloads\vision lang pipeline\vision-language-pipeline\frontend\observability-console\src\components\VideoAnalyzer.tsx'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix imports
text = re.sub(r'AlignLeft,', r'AlignLeft,\n  MessageSquare,', text)

# Fix interface
text = re.sub(r'interface VideoTrackerReport \{\n  event_detected: boolean;\n  summary: string;', r'interface VideoTrackerReport {\n  event_detected: boolean;\n  summary: string;\n  custom_prompt_response?: string;', text)

# Fix rendering
summary_block = r'''        {/* Summary */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
          <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-2 flex items-center gap-1.5">
            <AlignLeft className="w-3.5 h-3.5" /> Summary
          </p>
          <p className="text-slate-200 text-sm leading-relaxed">{report.summary}</p>
        </div>'''

new_summary_block = r'''        {/* Summary */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
          <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-2 flex items-center gap-1.5">
            <AlignLeft className="w-3.5 h-3.5" /> Summary
          </p>
          <p className="text-slate-200 text-sm leading-relaxed">{report.summary}</p>
        </div>

        {/* Custom Prompt Response */}
        {report.custom_prompt_response && (
          <div className="bg-indigo-900/40 rounded-xl border border-indigo-500/50 p-4 shadow-inner shadow-indigo-500/10">
            <p className="text-xs text-indigo-300 uppercase tracking-wide font-semibold mb-2 flex items-center gap-1.5">
              <MessageSquare className="w-3.5 h-3.5" /> AI Response to Prompt
            </p>
            <p className="text-slate-200 text-sm leading-relaxed">{report.custom_prompt_response}</p>
          </div>
        )}'''

if summary_block in text:
    text = text.replace(summary_block, new_summary_block)
else:
    print("WARNING: Summary block not found exactly. Trying regex.")
    text = re.sub(
        r'\{/\* Summary \*/\}.*?\{report\.summary\}</p>\s*</div>',
        new_summary_block,
        text,
        flags=re.DOTALL
    )

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Frontend patched.")
