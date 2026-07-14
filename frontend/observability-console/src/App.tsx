import React, { useState, useEffect, useRef, useCallback } from 'react';
import VideoAnalyzer from './components/VideoAnalyzer';
import axios from 'axios';
import {
  UploadCloud, Activity, CheckCircle, AlertTriangle, FileText,
  RefreshCw, XCircle, Zap, Shield, Clock, FileVideo, AlignLeft
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000/api/v1';

type JobStatus = {
  status: 'queued' | 'extracting' | 'validating' | 'healing' | 'completed' | 'failed' | 'rejected';
  parsed_data?: Record<string, unknown>;
  error_message?: string | null;
  retries?: number;
};

type LogEntry = {
  time: string;
  msg: string;
  level: 'info' | 'warn' | 'success' | 'error';
};

const statusMeta: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  queued:     { color: 'text-slate-400',  icon: <Clock className="w-4 h-4" />,        label: 'Queued'      },
  extracting: { color: 'text-blue-400',   icon: <RefreshCw className="w-4 h-4 animate-spin" />, label: 'Extracting' },
  validating: { color: 'text-yellow-400', icon: <Shield className="w-4 h-4" />,       label: 'Validating'  },
  healing:    { color: 'text-orange-400', icon: <AlertTriangle className="w-4 h-4" />, label: 'Self-Healing'},
  completed:  { color: 'text-emerald-400',icon: <CheckCircle className="w-4 h-4" />,  label: 'Completed'   },
  failed:     { color: 'text-red-400',    icon: <XCircle className="w-4 h-4" />,      label: 'Failed'      },
  rejected:   { color: 'text-purple-400', icon: <Shield className="w-4 h-4" />,       label: 'Rejected'    },
};

function now() {
  return new Date().toLocaleTimeString('en-US', { hour12: false });
}

type ActiveTab = 'invoice' | 'video';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('invoice');
  const [file, setFile]           = useState<File | null>(null);
  const [invoicePrompt, setInvoicePrompt] = useState('Extract all billing details, line items, quantities, unit prices, taxes, and payment terms precisely.');
  const [isDragging, setIsDragging] = useState(false);
  const [docId, setDocId]         = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [logs, setLogs]           = useState<LogEntry[]>([]);
  const logEndRef                 = useRef<HTMLDivElement>(null);
  const fileInputRef              = useRef<HTMLInputElement>(null);

  const pushLog = useCallback((msg: string, level: LogEntry['level'] = 'info') => {
    setLogs(prev => [...prev, { time: now(), msg, level }]);
  }, []);

  // Auto-scroll logs to bottom
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // ── Drag & Drop ──────────────────────────────────────────────────────────
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) { setFile(dropped); setJobStatus(null); setDocId(null); setLogs([]); }
  };
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (picked) { setFile(picked); setJobStatus(null); setDocId(null); setLogs([]); }
  };

  // ── Trigger Pipeline (sends real file bytes via multipart/form-data) ─────
  const startProcessing = async () => {
    if (!file || isPolling) return;
    setLogs([]);
    setJobStatus(null);
    try {
      pushLog(`Uploading "${file.name}" to pipeline gateway...`, 'info');

      // Build a FormData payload with the actual file and optional prompt
      const formData = new FormData();
      formData.append('file', file);
      if (invoicePrompt.trim()) formData.append('prompt', invoicePrompt.trim());

      const res = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const id: string = res.data.document_id;
      setDocId(id);
      setIsPolling(true);
      pushLog(`Job initialised — Document ID: ${id}`, 'info');
      pushLog('LangGraph orchestrator started. Polling for telemetry...', 'info');
    } catch (err) {
      pushLog('Failed to reach backend. Is the server running?', 'error');
    }
  };

  // ── Telemetry Polling ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!isPolling || !docId) return;
    let prevStatus = '';
    let prevRetries = -1;

    const interval = window.setInterval(async () => {
      try {
        const res = await axios.get<JobStatus>(`${API_BASE}/status/${docId}`);
        const data = res.data;
        setJobStatus(data);

        // Emit log lines only on state transitions
        if (data.status !== prevStatus) {
          const meta = statusMeta[data.status];
          pushLog(`[Node: ${meta?.label ?? data.status}] Graph transitioned → ${data.status.toUpperCase()}`,
            data.status === 'completed' ? 'success'
            : data.status === 'failed'  ? 'error'
            : data.status === 'healing' ? 'warn'
            : 'info');
          prevStatus = data.status;
        }
        if ((data.retries ?? 0) > prevRetries && (data.retries ?? 0) > 0) {
          pushLog(`Pydantic validation caught errors — self-healing triggered (attempt ${data.retries})`, 'warn');
          prevRetries = data.retries ?? 0;
        }
        if (data.status === 'completed') {
          pushLog('Extraction validated successfully. Parsed data persisted.', 'success');
          setIsPolling(false);
        }
        if (data.status === 'failed') {
          pushLog(`Pipeline exhausted max retries. Error: ${data.error_message ?? 'unknown'}`, 'error');
          setIsPolling(false);
        }
      } catch {
        pushLog('Polling error — retrying...', 'warn');
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isPolling, docId, pushLog]);

  const canSubmit = !!file && !isPolling;
  const currentMeta = jobStatus ? (statusMeta[jobStatus.status] ?? statusMeta.queued) : null;

  // If the video tab is active, render its dedicated component
  if (activeTab === 'video') {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
        {/* Tab switcher header for Video tab */}
        <div className="border-b border-slate-800 bg-slate-900 sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-1">
            <button
              onClick={() => setActiveTab('invoice')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-all duration-150"
            >
              <Zap className="w-4 h-4" /> Invoice Pipeline
            </button>
            <button
              onClick={() => setActiveTab('video')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-violet-600/20 text-violet-300 border border-violet-500/30"
            >
              <FileVideo className="w-4 h-4" /> Video Tracker
            </button>
          </div>
        </div>
        <VideoAnalyzer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col overflow-y-auto">

      {/* ── Top Navigation Bar ──────────────────────────────────────────── */}
      <nav className="border-b border-slate-800 bg-slate-900">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-600/20 border border-blue-500/30">
              <Zap className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">VLP Observability Console</h1>
              <p className="text-xs text-slate-500">Vision-Language Agentic Pipeline</p>
            </div>
          </div>
          {/* ── Tab Switcher ──────────────────────────────────────────────── */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveTab('invoice')}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium bg-blue-600/20 text-blue-300 border border-blue-500/30"
            >
              <Zap className="w-3.5 h-3.5" /> Invoice Pipeline
            </button>
            <button
              onClick={() => setActiveTab('video')}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-all duration-150"
            >
              <FileVideo className="w-3.5 h-3.5" /> Video Tracker
            </button>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-3 py-1 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              LangGraph Active
            </span>
          </div>
        </div>
      </nav>

      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8 grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── LEFT: Intake Gateway ─────────────────────────────────────────── */}
        <div className="flex flex-col gap-6">

          {/* Drop Zone */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <UploadCloud className="w-4 h-4 text-indigo-400" /> Intake Gateway
            </h2>

            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`relative rounded-xl border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-200
                ${isDragging
                  ? 'border-blue-500 bg-blue-500/10 scale-[1.01]'
                  : file
                    ? 'border-emerald-500/50 bg-emerald-500/5'
                    : 'border-slate-700 bg-slate-800/50 hover:border-slate-600 hover:bg-slate-800'}`}
            >
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileInput} />
              {file ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="p-3 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                    <FileText className="w-8 h-8 text-emerald-400" />
                  </div>
                  <div>
                    <p className="text-emerald-400 font-semibold">{file.name}</p>
                    <p className="text-slate-500 text-sm mt-1">{(file.size / 1024).toFixed(1)} KB · Ready for extraction</p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div className="p-3 rounded-full bg-slate-700/50 border border-slate-600">
                    <UploadCloud className="w-8 h-8 text-slate-400" />
                  </div>
                  <div>
                    <p className="text-slate-300 font-medium">Drop invoice image here</p>
                    <p className="text-slate-500 text-sm mt-1">or click to browse files</p>
                  </div>
                </div>
              )}
            </div>

            {/* ── Analysis Prompt ─────────────────────────────────────────── */}
            <div className="mt-4">
              <label
                htmlFor="invoice-prompt"
                className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1.5"
              >
                <AlignLeft className="w-3.5 h-3.5 text-blue-400" /> Analysis Prompt
              </label>
              <textarea
                id="invoice-prompt"
                value={invoicePrompt}
                onChange={e => setInvoicePrompt(e.target.value)}
                disabled={isPolling}
                rows={3}
                placeholder="Describe what to focus on during extraction..."
                className={`w-full bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/30 transition-colors duration-200 ${isPolling ? 'opacity-50 cursor-not-allowed' : ''}`}
              />
              <p className="text-xs text-slate-600 mt-1.5">
                Guides Gemini's extraction focus. Combined with the base system instruction.
              </p>
            </div>

            <button
              onClick={startProcessing}
              disabled={!canSubmit}
              className={`mt-4 w-full py-3 px-6 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2
                ${canSubmit
                  ? 'bg-blue-600 hover:bg-blue-500 active:scale-[0.98] text-white shadow-lg shadow-blue-500/20'
                  : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'}`}
            >
              {isPolling
                ? <><RefreshCw className="w-4 h-4 animate-spin" /> Orchestrating Pipeline...</>
                : <><Zap className="w-4 h-4" /> Trigger VLM Pipeline</>}
            </button>
          </div>

          {/* Status Card */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400" /> Current Job
            </h2>
            {!docId ? (
              <p className="text-slate-600 text-sm italic">No active job.</p>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500 uppercase tracking-wide">Document ID</span>
                  <code className="text-xs text-slate-300 bg-slate-800 px-2 py-1 rounded font-mono">
                    {docId.slice(0, 8)}…
                  </code>
                </div>
                {currentMeta && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500 uppercase tracking-wide">Status</span>
                    <span className={`flex items-center gap-1.5 text-sm font-medium ${currentMeta.color}`}>
                      {currentMeta.icon} {currentMeta.label}
                    </span>
                  </div>
                )}
                {(jobStatus?.retries ?? 0) > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500 uppercase tracking-wide">Self-Heals</span>
                    <span className="text-orange-400 text-sm font-medium">
                      {jobStatus?.retries} attempt{(jobStatus?.retries ?? 0) > 1 ? 's' : ''}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── RIGHT: Telemetry + Parsed Output ────────────────────────────── */}
        <div className="flex flex-col gap-6">

          {/* Live Log Window */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 flex flex-col" style={{ minHeight: '260px' }}>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-green-400" /> Pipeline Telemetry
              {isPolling && <span className="ml-auto text-xs text-blue-400 flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" /> Live</span>}
            </h2>
            <div className="flex-1 bg-black/60 rounded-xl p-4 font-mono text-xs overflow-y-auto border border-slate-800 space-y-1.5 min-h-[160px]">
              {logs.length === 0
                ? <span className="text-slate-600">Waiting for job initialization...</span>
                : logs.map((l, i) => (
                  <div key={i} className={`flex gap-2
                    ${l.level === 'success' ? 'text-emerald-400'
                    : l.level === 'error'   ? 'text-red-400'
                    : l.level === 'warn'    ? 'text-orange-400'
                    : 'text-slate-400'}`}>
                    <span className="text-slate-600 shrink-0">[{l.time}]</span>
                    <span>{l.msg}</span>
                  </div>
                ))}
              <div ref={logEndRef} />
            </div>
          </div>

          {/* Parsed Data Output */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 flex flex-col flex-1">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-emerald-400" /> Extraction Output
            </h2>
            <div className="flex-1 bg-black/60 rounded-xl border border-slate-800 overflow-hidden">
              {jobStatus?.status === 'completed' && jobStatus.parsed_data ? (
                <pre className="text-emerald-300 text-xs p-4 overflow-auto h-full leading-relaxed">
                  {JSON.stringify(jobStatus.parsed_data, null, 2)}
                </pre>
              ) : jobStatus?.status === 'failed' ? (
                <div className="p-4 text-red-400 text-xs font-mono flex items-start gap-2">
                  <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{jobStatus.error_message ?? 'Pipeline failed after maximum retries.'}</span>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full p-8">
                  <p className="text-slate-700 text-sm italic">
                    {isPolling ? 'Awaiting validated extraction...' : 'Trigger the pipeline to see output.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
