import React, { useState, useRef, useCallback } from 'react';
import {
  UploadCloud, FileVideo, Loader2, AlertTriangle,
  CheckCircle2, XCircle, Scan, Crosshair, Clock,
  AlignLeft,
  MessageSquare, Send, RefreshCw, Boxes
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────
// Types — mirror of server.py VideoTrackerReport Pydantic schema
// ─────────────────────────────────────────────────────────────────────────────

interface TrackedObject {
  label: string;
  timestamp_seconds: number | null;
  box_2d: number[] | null; // [ymin, xmin, ymax, xmax] normalized 0.0–1.0
}

interface VideoTrackerReport {
  event_detected: boolean;
  summary: string;
  custom_prompt_response?: string;
  custom_prompt_response?: string;
  tracked_objects: TrackedObject[];
}

type AnalysisState =
  | { phase: 'idle' }
  | { phase: 'uploading' }
  | { phase: 'processing' }
  | { phase: 'polling'; jobId: string; attempts: number }
  | { phase: 'success'; report: VideoTrackerReport }
  | { phase: 'error'; message: string };

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE      = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const VIDEO_API_URL = `${API_BASE}/api/v1/analyze-video`;
const JOB_URL       = (jobId: string) => `${API_BASE}/api/v1/job/${jobId}`;
const API_KEY       = import.meta.env.VITE_API_KEY ?? '';  // set in .env.local for dev
const POLL_INTERVAL_MS  = 3000;   // Poll every 3 s
const MAX_POLL_ATTEMPTS = 60;     // Give up after 3 min (60 × 3 s)
const ACCEPTED_TYPES = new Set(['.mp4', '.mov', '.avi', '.mkv', '.webm']);
const DEFAULT_PROMPT = 'Detect and track all objects, people, and events visible in this video.';

function formatTimestamp(ts: number | null): string {
  if (ts === null || ts === undefined) return '—';
  const mins = Math.floor(ts / 60);
  const secs = (ts % 60).toFixed(1).padStart(4, '0');
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

function formatBox2d(box: number[] | null): string {
  if (!box || box.length < 4) return '—';
  return `[${box.map(v => v.toFixed(3)).join(', ')}]`;
}

function fileExtension(name: string): string {
  return name.slice(name.lastIndexOf('.')).toLowerCase();
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function ProcessingOverlay({ phase }: { phase: 'uploading' | 'processing' | 'polling' }) {
  const label =
    phase === 'uploading' ? 'Uploading to server...' :
    phase === 'polling'   ? 'Waiting for Gemini...' :
                           'Gemini analyzing video...';
  const sublabel =
    phase === 'uploading' ? 'Streaming video bytes to the API gateway.' :
    phase === 'polling'   ? 'Polling for results every 3 s. This may take 15–60 seconds.' :
                           'Gemini 2.5 Flash is processing frames. This may take 15–60 seconds.';

  return (
    <div className="flex flex-col items-center justify-center gap-5 py-12 px-6 text-center">
      <div className="relative">
        <div className="w-16 h-16 rounded-full border-4 border-slate-700" />
        <div className="absolute inset-0 w-16 h-16 rounded-full border-4 border-t-blue-500 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          {phase === 'uploading'
            ? <UploadCloud className="w-6 h-6 text-blue-400" />
            : <Scan className="w-6 h-6 text-violet-400 animate-pulse" />
          }
        </div>
      </div>
      <div>
        <p className="text-slate-200 font-semibold text-base">{label}</p>
        <p className="text-slate-500 text-sm mt-1 max-w-xs">{sublabel}</p>
      </div>
      {(phase === 'processing' || phase === 'polling') && (
        <div className="flex items-center gap-1.5 text-xs text-violet-400 bg-violet-400/10 border border-violet-400/20 px-3 py-1.5 rounded-full">
          <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
          Cloud inference active — gemini-2.5-flash
        </div>
      )}
    </div>
  );
}

function EventBadge({ detected }: { detected: boolean }) {
  if (detected) {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/25">
        <CheckCircle2 className="w-4 h-4" />
        Event Detected
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-slate-700/60 text-slate-400 border border-slate-600/40">
      <XCircle className="w-4 h-4" />
      No Event Detected
    </span>
  );
}

function TrackedObjectsTable({ objects }: { objects: TrackedObject[] }) {
  if (objects.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-slate-600 gap-2">
        <Boxes className="w-8 h-8" />
        <p className="text-sm italic">No objects tracked in this video.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-800/70 border-b border-slate-700/60">
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              # 
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              <span className="flex items-center gap-1.5"><Scan className="w-3.5 h-3.5" /> Label</span>
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              <span className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> Timestamp</span>
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              <span className="flex items-center gap-1.5"><Crosshair className="w-3.5 h-3.5" /> Bounding Box (ymin, xmin, ymax, xmax)</span>
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {objects.map((obj, idx) => (
            <tr
              key={idx}
              className="bg-slate-900/50 hover:bg-slate-800/40 transition-colors duration-150"
            >
              <td className="px-4 py-3 text-slate-600 font-mono text-xs">{idx + 1}</td>
              <td className="px-4 py-3">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-blue-500/10 border border-blue-500/20 text-blue-300 font-medium text-xs">
                  {obj.label}
                </span>
              </td>
              <td className="px-4 py-3 font-mono text-xs text-slate-300">
                {formatTimestamp(obj.timestamp_seconds)}
              </td>
              <td className="px-4 py-3 font-mono text-xs text-slate-400">
                {formatBox2d(obj.box_2d)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReportDisplay({ report }: { report: VideoTrackerReport }) {
  return (
    <div className="flex flex-col gap-5 animate-in fade-in duration-500">

      {/* Event Detected Banner */}
      <div className={`rounded-xl border p-4 flex items-center justify-between
        ${report.event_detected
          ? 'bg-emerald-950/30 border-emerald-500/25'
          : 'bg-slate-800/40 border-slate-700/40'
        }`}
      >
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-1">
            Detection Result
          </p>
          <EventBadge detected={report.event_detected} />
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-1">
            Objects Tracked
          </p>
          <p className="text-2xl font-bold text-slate-200">
            {report.tracked_objects.length}
          </p>
        </div>
      </div>

              {/* Summary */}
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
        )}

      {/* Tracked Objects Table */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
        <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-3 flex items-center gap-1.5">
          <Crosshair className="w-3.5 h-3.5" /> Tracked Objects
        </p>
        <TrackedObjectsTable objects={report.tracked_objects} />
      </div>

      {/* Raw JSON Accordion */}
      <details className="group bg-slate-900 rounded-xl border border-slate-800">
        <summary className="px-4 py-3 cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500 flex items-center gap-2 select-none hover:text-slate-400 transition-colors">
          <span className="group-open:rotate-90 transition-transform duration-200 inline-block">▶</span>
          Raw JSON Response
        </summary>
        <pre className="px-4 pb-4 text-xs text-emerald-300 leading-relaxed overflow-x-auto">
          {JSON.stringify(report, null, 2)}
        </pre>
      </details>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

export default function VideoAnalyzer() {
  const [file, setFile]           = useState<File | null>(null);
  const [videoUrl, setVideoUrl]   = useState<string | null>(null);
  const [prompt, setPrompt]       = useState(DEFAULT_PROMPT);
  const [isDragging, setIsDragging] = useState(false);
  const [state, setState]         = useState<AnalysisState>({ phase: 'idle' });
  const fileInputRef              = useRef<HTMLInputElement>(null);

  const isProcessing = state.phase === 'uploading' || state.phase === 'processing' || state.phase === 'polling';

  // ── File selection helpers ─────────────────────────────────────────────────
  const acceptFile = useCallback((picked: File) => {
    const ext = fileExtension(picked.name);
    if (!ACCEPTED_TYPES.has(ext)) {
      setState({ phase: 'error', message: `Unsupported file type "${ext}". Please upload an .mp4, .mov, .avi, .mkv, or .webm file.` });
      return;
    }
    // Revoke any previous object URL before creating a new one
    setVideoUrl(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(picked); });
    setFile(picked);
    setState({ phase: 'idle' });
  }, []);

  const handleDragOver  = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop      = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) acceptFile(dropped);
  };
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (picked) acceptFile(picked);
  };

  // ── Reset handler ──────────────────────────────────────────────────────────
  const handleReset = () => {
    // Revoke the object URL to free browser memory before clearing state
    setVideoUrl(prev => { if (prev) URL.revokeObjectURL(prev); return null; });
    setFile(null);
    setPrompt(DEFAULT_PROMPT);
    setState({ phase: 'idle' });
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ── Submit handler ─────────────────────────────────────────────────────────
  // Step 1: POST the file → backend returns 202 + job_id immediately.
  // Step 2: Poll GET /api/v1/job/{job_id} every 3 s until status is
  //         'completed' or 'failed' (or MAX_POLL_ATTEMPTS is exhausted).
  const handleSubmit = async () => {
    if (!file || isProcessing) return;

    setState({ phase: 'uploading' });

    const formData = new FormData();
    formData.append('file', file);
    if (prompt.trim()) formData.append('prompt', prompt.trim());

    // ── Step 1: Upload the file and receive a job_id ──────────────────────
    let jobId: string;
    try {
      const uploadRes = await fetch(VIDEO_API_URL, {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (uploadRes.status === 202) {
        const { job_id } = await uploadRes.json();
        jobId = job_id;
      } else {
        const body = await uploadRes.json().catch(() => ({ detail: uploadRes.statusText }));
        const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
        setState({ phase: 'error', message: `Upload failed ${uploadRes.status}: ${detail}` });
        return;
      }
    } catch {
      setState({ phase: 'error', message: 'Network error. Is the backend running?' });
      return;
    }

    // Upload is done — video is now in the background worker
    setState({ phase: 'polling', jobId, attempts: 0 });

    // ── Step 2: Poll until the job finishes ───────────────────────────────
    let attempts = 0;
    const intervalId = setInterval(async () => {
      attempts++;

      if (attempts > MAX_POLL_ATTEMPTS) {
        clearInterval(intervalId);
        setState({ phase: 'error', message: 'Timed out waiting for Gemini to finish. Please try again.' });
        return;
      }

      try {
        const pollRes = await fetch(JOB_URL(jobId));
        if (!pollRes.ok) {
          clearInterval(intervalId);
          setState({ phase: 'error', message: `Polling error ${pollRes.status}: ${pollRes.statusText}` });
          return;
        }

        const job = await pollRes.json();

        if (job.status === 'completed') {
          clearInterval(intervalId);
          setState({ phase: 'success', report: job.result as VideoTrackerReport });

        } else if (job.status === 'failed') {
          clearInterval(intervalId);
          setState({ phase: 'error', message: job.error ?? 'The analysis job failed on the server.' });
        }
        // 'pending' or 'processing' → keep polling
      } catch {
        // Don't abort on a single failed poll — network hiccup, try again next tick
        console.warn(`[Poll attempt ${attempts}] Network error, will retry...`);
      }
    }, POLL_INTERVAL_MS);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-violet-600/20 border border-violet-500/30">
              <FileVideo className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">
                Video Analytics Tracker
              </h1>
              <p className="text-xs text-slate-500">
                Gemini 2.5 Flash · Hybrid Cloud-Edge · Port 8001
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs text-violet-400 bg-violet-400/10 border border-violet-400/20 px-3 py-1 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
              Cloud Inference Ready
            </span>
          </div>
        </div>
      </header>

      {/* ── Main Layout ────────────────────────────────────────────────────── */}
      <main className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-5 gap-6">

        {/* ── LEFT COLUMN: Upload + Prompt (2 of 5 cols) ─────────────────── */}
        <div className="lg:col-span-2 flex flex-col gap-5">

          {/* Upload Zone */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-5">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <UploadCloud className="w-3.5 h-3.5 text-violet-400" /> Video Upload
            </h2>

            {/* Drop Zone */}
            <div
              onClick={() => !isProcessing && fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`
                relative rounded-xl border-2 border-dashed
                transition-all duration-200
                ${isProcessing ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}
                ${isDragging
                  ? 'border-violet-500 bg-violet-500/10 scale-[1.01]'
                  : file
                    ? 'border-emerald-500/40 bg-emerald-950/10'
                    : 'border-slate-700 bg-slate-800/40 hover:border-slate-600 hover:bg-slate-800/60'
                }
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm"
                className="hidden"
                onChange={handleFileInput}
                disabled={isProcessing}
              />

              {/* ── Video Preview — shown as soon as a file is selected ─── */}
              {videoUrl && file ? (
                <div className="flex flex-col">
                  {/* Native HTML5 video player */}
                  <video
                    key={videoUrl}
                    src={videoUrl}
                    controls
                    muted
                    className="w-full rounded-t-xl max-h-48 object-cover bg-black"
                    onClick={e => e.stopPropagation()} // prevent click from re-opening file picker
                  />
                  {/* File metadata strip below the player */}
                  <div className="flex items-center justify-between px-3 py-2.5">
                    <div className="flex items-center gap-2 min-w-0">
                      <FileVideo className="w-4 h-4 text-emerald-400 shrink-0" />
                      <p className="text-emerald-400 font-medium text-xs truncate">{file.name}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0 ml-2">
                      <span className="text-slate-500 text-xs">{formatFileSize(file.size)}</span>
                      <span className="text-slate-600 text-xs">·</span>
                      <span className="text-slate-500 text-xs">{fileExtension(file.name).toUpperCase()}</span>
                      {!isProcessing && (
                        <span className="text-slate-600 text-xs italic ml-1">click to change</span>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                /* ── Empty state — no file selected yet ───────────────── */
                <div className="flex flex-col items-center gap-3 p-8 text-center">
                  <div className="p-3 rounded-full bg-slate-700/60 border border-slate-600">
                    <UploadCloud className="w-7 h-7 text-slate-400" />
                  </div>
                  <div>
                    <p className="text-slate-300 font-medium text-sm">Drop video here</p>
                    <p className="text-slate-500 text-xs mt-1">or click to browse</p>
                    <p className="text-slate-600 text-xs mt-2">MP4 · MOV · AVI · MKV · WEBM</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Prompt Input */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-5">
            <label
              htmlFor="analysis-prompt"
              className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2"
            >
              <AlignLeft className="w-3.5 h-3.5 text-blue-400" /> Analysis Prompt
            </label>
            <textarea
              id="analysis-prompt"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              disabled={isProcessing}
              rows={4}
              placeholder="Describe what to focus on..."
              className={`
                w-full bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-3
                text-sm text-slate-200 placeholder-slate-600
                resize-none focus:outline-none focus:border-violet-500/60 focus:ring-1 focus:ring-violet-500/30
                transition-colors duration-200
                ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            />
            <p className="text-xs text-slate-600 mt-2">
              This is injected as an additional focus instruction alongside the base schema prompt.
            </p>
          </div>

          {/* Submit / Reset Buttons */}
          <div className="flex flex-col gap-2">
            <button
              onClick={handleSubmit}
              disabled={!file || isProcessing}
              className={`
                w-full py-3 px-6 rounded-xl font-semibold text-sm
                transition-all duration-200 flex items-center justify-center gap-2
                ${file && !isProcessing
                  ? 'bg-violet-600 hover:bg-violet-500 active:scale-[0.98] text-white shadow-lg shadow-violet-600/25'
                  : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                }
              `}
            >
              {isProcessing
                ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
                : <><Send className="w-4 h-4" /> Analyze Video</>
              }
            </button>

            {(file || state.phase !== 'idle') && (
              <button
                onClick={handleReset}
                disabled={isProcessing}
                className="w-full py-2.5 px-6 rounded-xl font-medium text-sm text-slate-500 border border-slate-800 hover:border-slate-700 hover:text-slate-400 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Reset
              </button>
            )}
          </div>
        </div>

        {/* ── RIGHT COLUMN: Results Panel (3 of 5 cols) ──────────────────── */}
        <div className="lg:col-span-3">
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-5 min-h-[500px] flex flex-col">

            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-5 flex items-center gap-2">
              <Crosshair className="w-3.5 h-3.5 text-emerald-400" /> Analysis Report
            </h2>

            {/* State Renderer */}
            {state.phase === 'idle' && (
              <div className="flex flex-col items-center justify-center flex-1 gap-4 text-center py-12">
                <div className="p-5 rounded-full bg-slate-800/60 border border-slate-700/50">
                  <FileVideo className="w-10 h-10 text-slate-600" />
                </div>
                <div>
                  <p className="text-slate-500 font-medium text-sm">No analysis yet</p>
                  <p className="text-slate-600 text-xs mt-1 max-w-xs">
                    Upload a video and click "Analyze Video" to get a structured tracker report from Gemini.
                  </p>
                </div>
              </div>
            )}

            {(state.phase === 'uploading' || state.phase === 'processing' || state.phase === 'polling') && (
              <div className="flex flex-col items-center justify-center flex-1">
                <ProcessingOverlay phase={state.phase === 'polling' ? 'polling' : state.phase} />
              </div>
            )}

            {state.phase === 'error' && (
              <div className="flex flex-col items-center gap-4 py-10 text-center flex-1">
                <div className="p-4 rounded-full bg-red-500/10 border border-red-500/25">
                  <AlertTriangle className="w-8 h-8 text-red-400" />
                </div>
                <div>
                  <p className="text-red-400 font-semibold text-base">Analysis Failed</p>
                  <p className="text-slate-500 text-sm mt-2 max-w-sm leading-relaxed">
                    {state.message}
                  </p>
                </div>
              </div>
            )}

            {state.phase === 'success' && (
              <div className="flex flex-col gap-1">
                {/* Success Banner */}
                <div className="flex items-center gap-2 mb-4 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-2 rounded-lg">
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                  <span>
                    Report validated locally against <code className="font-mono">VideoTrackerReport</code> schema.
                  </span>
                </div>
                <ReportDisplay report={state.report} />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
