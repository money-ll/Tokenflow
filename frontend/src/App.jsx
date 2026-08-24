import React, { useState } from "react";
import {
  Upload,
  FileText,
  FileType,
  Image as ImageIcon,
  History,
  CheckCircle2,
  Copy,
  Download,
  RotateCcw,
  AlertCircle,
  PenTool,
  FileSearch,
  Camera,
  EyeOff,
  Gauge,
} from "lucide-react";
import { optimizeFile, getHistory } from "./services/api";

const SOURCE_LABELS = {
  handwritten: { icon: PenTool, label: "Handwritten" },
  printed: { icon: FileSearch, label: "Printed / scanned" },
  mixed: { icon: FileSearch, label: "Mixed text + photo" },
  contextual: { icon: Camera, label: "Contextual image" },
  photo: { icon: Camera, label: "Photo, no text" },
  blank: { icon: EyeOff, label: "No text detected" },
  pdf: { icon: FileType, label: "PDF" },
  text: { icon: FileText, label: "Text file" },
};

function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("");
  const [targetReduction, setTargetReduction] = useState(0.45);
  const [evaluateQuality, setEvaluateQuality] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");

  const handleFile = (selected) => {
    const next = selected?.[0];
    if (!next) return;
    const ext = next.name.toLowerCase().split(".").pop();
    if (!["txt", "pdf", "png", "jpg", "jpeg"].includes(ext)) {
      setError("Please choose a .txt, .pdf, or image file (.png / .jpg).");
      return;
    }
    if (next.size > 10 * 1024 * 1024) {
      setError("Maximum file size is 10 MB.");
      return;
    }
    setError("");
    setFile(next);
    setResult(null);
  };

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      setResult(await optimizeFile(file, query, targetReduction, evaluateQuality));
      refreshHistory();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const refreshHistory = async () => {
    try {
      const data = await getHistory();
      setHistory(data.items || []);
    } catch (e) {
      // ignore silent refresh failures
    }
  };

  const openHistory = async () => {
    setShowHistory(true);
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const data = await getHistory();
      setHistory(data.items || []);
    } catch (e) {
      setHistoryError("Couldn't load history.");
    } finally {
      setHistoryLoading(false);
    }
  };

  const copy = async () => {
    if (result?.assembled_prompt) {
      await navigator.clipboard.writeText(result.assembled_prompt);
    }
  };

  const download = () => {
    if (!result) return;
    const blob = new Blob([result.assembled_prompt], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${result.filename}.optimized.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const sourceInfo = result?.source ? SOURCE_LABELS[result.source.source_type] : null;

  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">TokenFlow</span>
        <nav>
          <button onClick={openHistory}>
            <History size={15} /> History
          </button>
        </nav>
      </header>

      {showHistory && (
        <div className="history-overlay" onClick={() => setShowHistory(false)}>
          <div className="history-panel" onClick={(e) => e.stopPropagation()}>
            <div className="history-panel-header">
              <h3>Recent runs</h3>
              <button className="history-close" onClick={() => setShowHistory(false)}>
                ×
              </button>
            </div>

            {historyLoading && <p className="history-status">Loading…</p>}
            {historyError && <p className="history-status error">{historyError}</p>}
            {!historyLoading && !historyError && history.length === 0 && (
              <p className="history-status">No history yet.</p>
            )}

            <div className="history-list">
              {history.map((item) => (
                <div className="history-item" key={item.id}>
                  <div className="history-item-main">
                    <b>{item.filename}</b>
                    <span className="history-item-date">
                      {item.created_at ? new Date(item.created_at).toLocaleString() : ""}
                    </span>
                  </div>
                  <span className="history-item-reduction">
                    {item.metrics?.token_reduction_rate ?? "–"}% reduced
                    {item.evaluation?.bertscore_f1 != null && (
                      <> · F1 {item.evaluation.bertscore_f1}</>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <main className="container">
        {!result ? (
          <section className="workspace">
            <div
              className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                handleFile(e.dataTransfer.files);
              }}
              onClick={() => document.getElementById("file-input").click()}
            >
              <input
                id="file-input"
                type="file"
                accept=".txt,.pdf,.png,.jpg,.jpeg"
                hidden
                onChange={(e) => handleFile(e.target.files)}
              />
              <Upload size={22} className="upload-icon" />
              {file ? (
                <>
                  <p className="dropzone-name">{file.name}</p>
                  <p className="dropzone-sub">{(file.size / 1024).toFixed(1)} KB</p>
                </>
              ) : (
                <>
                  <p className="dropzone-name">Drop a file, or click to browse</p>
                  <p className="dropzone-sub">.txt, .pdf, .png, .jpg — up to 10 MB</p>
                </>
              )}
            </div>

            <div className="controls">
              <label>
                <span>Task / query (optional)</span>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Introduction"
                />
              </label>
              <label>
                <span>
                  Compression target — <b>{Math.round(targetReduction * 100)}%</b>
                </span>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={targetReduction}
                  onChange={(e) => setTargetReduction(Number(e.target.value))}
                />
                {/* <small>Negations and meaning-critical words are always preserved.</small> */}
              </label>
            </div>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={evaluateQuality}
                onChange={(e) => setEvaluateQuality(e.target.checked)}
              />
              <span>
                Evaluate quality (BERTScore)
              </span>
            </label>

            {error && <div className="error">{error}</div>}

            <button className="primary-btn" disabled={!file || loading} onClick={run}>
              {loading ? (
                <>
                  <span className="spinner"></span>{" "}
                  {evaluateQuality ? "Optimizing & evaluating…" : "Optimizing…"}
                </>
              ) : (
                "Optimize"
              )}
            </button>
          </section>
        ) : (
          <section className="results">
            <div className="result-header">
              <div>
                <h2>{result.filename}</h2>
                <div className="source-meta">
                  {sourceInfo && (
                    <span className="source-badge">
                      <sourceInfo.icon size={13} />
                      {sourceInfo.label}
                    </span>
                  )}
                  {result.section?.found && (
                    <span className="source-badge">
                      {result.section.requested} section only
                    </span>
                  )}
                </div>

                {result.section && !result.section.found && (
                  <div className="image-description">
                    <AlertCircle size={14} />
                    <span>
                      Couldn't find a "{result.section.requested}" section in this
                      document — optimized the full text instead.
                    </span>
                  </div>
                )}

                {result.source?.source_type === "pdf" && (
                  <p className="pdf-page-summary">
                    {result.source.typed_pages} typed page
                    {result.source.typed_pages === 1 ? "" : "s"}
                    {result.source.ocr_pages > 0 && (
                      <> · {result.source.ocr_pages} scanned page{result.source.ocr_pages === 1 ? "" : "s"} via OCR</>
                    )}
                    {result.source.blank_pages > 0 && (
                      <> · {result.source.blank_pages} blank skipped</>
                    )}
                  </p>
                )}

                {result.source?.description && !result.source?.has_text && (
                  <div className="image-description">
                    <AlertCircle size={14} />
                    <span>{result.source.description}</span>
                  </div>
                )}
              </div>

              <div className="actions">
                <button onClick={copy}>
                  <Copy size={14} /> Copy
                </button>
                <button onClick={download}>
                  <Download size={14} /> Download
                </button>
                <button
                  onClick={() => {
                    setResult(null);
                    setFile(null);
                  }}
                >
                  <RotateCcw size={14} /> New
                </button>
              </div>
            </div>

            <div className="metrics">
              <Metric label="Token reduction" value={`${result.metrics.token_reduction_rate}%`} accent />
              <Metric label="Original tokens" value={result.metrics.original_tokens.toLocaleString()} />
              <Metric label="Optimized tokens" value={result.metrics.optimized_tokens.toLocaleString()} />
              <Metric label="Duplicates removed" value={result.metrics.duplicate_sentences_removed} />
            </div>

            {result.evaluation && (
              <div className="panel eval-panel">
                <div className="panel-title">
                  <span>
                  Quality evaluation
                  </span>
                </div>
                {result.evaluation.error ? (
                  <p className="eval-error">{result.evaluation.error}</p>
                ) : (
                  <div className="eval-grid">
                    <Metric label="BERTScore F1" value={result.evaluation.bertscore_f1} accent />
                  </div>
                )}
              </div>
            )}

            <div className="result-grid">
              <div className="panel">
                <div className="panel-title">
                  Optimized prompt <span>{result.metrics.optimized_tokens} tokens</span>
                </div>
                <pre>{result.assembled_prompt}</pre>
              </div>

              <div className="panel">
                <div className="panel-title">Pipeline</div>
                <div className="stage-list">
                  {Object.entries(result.metrics.stage_tokens).map(([name, value]) => (
                    <div className="stage" key={name}>
                      <span>{name.replaceAll("_", " ")}</span>
                      <b>{value.toLocaleString()}</b>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function Metric({ label, value, accent }) {
  return (
    <div className={`metric ${accent ? "accent" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
