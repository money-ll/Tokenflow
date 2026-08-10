import React, { useState } from "react";
import {
  Upload,
  FileText,
  FileType,
  Image as ImageIcon,
  Zap,
  BarChart3,
  History,
  Settings,
  CheckCircle2,
  Copy,
  Download,
  RotateCcw,
  AlertCircle,
  Camera,
  PenTool,
  FileSearch,
  EyeOff,
} from "lucide-react";
import { optimizeFile } from "./services/api";

function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("");
  const [targetReduction, setTargetReduction] = useState(0.45);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

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
      setResult(await optimizeFile(file, query, targetReduction));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
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

  // ---------- helpers for image metadata ----------
  const getSourceBadge = (source) => {
    if (!source) return null;

    const type = source.source_type;

    const map = {
      handwritten: {
        icon: <PenTool size={14} />,
        label: "Handwritten",
        className: "badge-handwritten",
      },
      printed: {
        icon: <FileSearch size={14} />,
        label: "Printed / Scanned",
        className: "badge-printed",
      },
      mixed: {
        icon: <FileSearch size={14} />,
        label: "Mixed (text + photo)",
        className: "badge-mixed",
      },
      contextual: {
        icon: <Camera size={14} />,
        label: "Contextual image",
        className: "badge-contextual",
      },
      photo: {
        icon: <Camera size={14} />,
        label: "Photo (no text)",
        className: "badge-photo",
      },
      blank: {
        icon: <EyeOff size={14} />,
        label: "No text detected",
        className: "badge-blank",
      },
      pdf: {
        icon: <FileType size={14} />,
        label: "PDF",
        className: "badge-pdf",
      },
      text: {
        icon: <FileText size={14} />,
        label: "Text file",
        className: "badge-text",
      },
    };

    const info = map[type] || {
      icon: <ImageIcon size={14} />,
      label: type || "Unknown",
      className: "badge-default",
    };

    return (
      <span className={`source-badge ${info.className}`}>
        {info.icon}
        {info.label}
        {typeof source.confidence === "number" && source.has_text && (
          <span className="confidence">
            {Math.round(source.confidence * 100)}% conf
          </span>
        )}
      </span>
    );
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Zap size={17} />
          </div>
          <span>TokenFlow</span>
        </div>
        <nav>
          <button>
            <BarChart3 size={16} /> Analytics
          </button>
          <button>
            <History size={16} /> History
          </button>
          <button>
            <Settings size={16} /> Settings
          </button>
        </nav>
      </header>

      <main className="container">
        <section className="hero">
          <div className="eyebrow">
            <span className="pulse"></span> Semantic-aware token optimization
          </div>
          <h1>
            Optimize Your <span>LLM Tokens</span>
          </h1>
          <p>
            Reduce token consumption while preserving semantic integrity.
            Supports text, PDFs, handwritten notes, scanned pages and photos.
          </p>
        </section>

        {!result ? (
          <section className="workspace">
            <div
              className={`dropzone ${dragging ? "dragging" : ""} ${
                file ? "has-file" : ""
              }`}
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
              <div className="upload-icon">
                <Upload size={28} />
              </div>
              {file ? (
                <>
                  <h2>{file.name}</h2>
                  <p>
                    {(file.size / 1024).toFixed(1)} KB · Ready to optimize
                  </p>
                  <div className="file-types">
                    <span>
                      <FileText size={15} /> TXT
                    </span>
                    <span>
                      <FileType size={15} /> PDF
                    </span>
                    <span>
                      <ImageIcon size={15} /> Images
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <h2>Upload Your Document</h2>
                  <p>
                    Drag & drop or click · .txt, .pdf, handwritten notes,
                    scanned pages or photos (max 10 MB)
                  </p>
                  <div className="file-types">
                    <span>
                      <FileText size={15} /> .txt
                    </span>
                    <span>
                      <FileType size={15} /> .pdf
                    </span>
                    <span>
                      <ImageIcon size={15} /> .png / .jpg
                    </span>
                  </div>
                </>
              )}
            </div>

            <div className="controls">
              <label>
                <span>Optional task / query</span>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Example: Summarize the key findings and preserve all numbers..."
                />
              </label>
              <label>
                <span>
                  Compression target <b>{Math.round(targetReduction * 100)}%</b>
                </span>
                <input
                  type="range"
                  min="0.15"
                  max="0.75"
                  step="0.05"
                  value={targetReduction}
                  onChange={(e) => setTargetReduction(Number(e.target.value))}
                />
                <small>
                  Higher compression is more aggressive. Negations and
                  meaning-critical words are protected.
                </small>
              </label>
            </div>

            {error && <div className="error">{error}</div>}

            <button
              className="primary-btn"
              disabled={!file || loading}
              onClick={run}
            >
              {loading ? (
                <>
                  <span className="spinner"></span> Optimizing...
                </>
              ) : (
                <>
                  <Zap size={18} /> Optimize Tokens
                </>
              )}
            </button>
          </section>
        ) : (
          <section className="results">
            <div className="result-header">
              <div>
                <div className="eyebrow">
                  <CheckCircle2 size={15} />
                  {result.source?.source_type === "handwritten"
                    ? "Handwriting recognized & optimized"
                    : result.source?.source_type === "photo" ||
                      result.source?.source_type === "blank"
                    ? "Image analyzed"
                    : "Optimization complete"}
                </div>
                <h2>{result.filename}</h2>

                {/* Source type badge + confidence */}
                <div className="source-meta">
                  {getSourceBadge(result.source)}
                </div>

                {/* PDF page summary */}
                {result.source?.source_type === "pdf" && (
                  <p className="pdf-page-summary">
                    {result.source.typed_pages} typed page
                    {result.source.typed_pages === 1 ? "" : "s"}
                    {result.source.ocr_pages > 0 && (
                      <>
                        {" "}
                        · {result.source.ocr_pages} scanned page
                        {result.source.ocr_pages === 1 ? "" : "s"} recovered via
                        OCR
                      </>
                    )}
                    {result.source.blank_pages > 0 && (
                      <>
                        {" "}
                        · {result.source.blank_pages} blank page
                        {result.source.blank_pages === 1 ? "" : "s"} skipped
                      </>
                    )}
                  </p>
                )}

                {/* Image description when no useful text */}
                {result.source?.description && !result.source?.has_text && (
                  <div className="image-description">
                    <AlertCircle size={15} />
                    <span>{result.source.description}</span>
                  </div>
                )}
              </div>

              <div className="actions">
                <button onClick={copy}>
                  <Copy size={16} /> Copy
                </button>
                <button onClick={download}>
                  <Download size={16} /> Download
                </button>
                <button
                  onClick={() => {
                    setResult(null);
                    setFile(null);
                  }}
                >
                  <RotateCcw size={16} /> New
                </button>
              </div>
            </div>

            <div className="metrics">
              <Metric
                label="Token Reduction"
                value={`${result.metrics.token_reduction_rate}%`}
                accent
              />
              <Metric
                label="Original Tokens"
                value={result.metrics.original_tokens.toLocaleString()}
              />
              <Metric
                label="Optimized Tokens"
                value={result.metrics.optimized_tokens.toLocaleString()}
              />
              <Metric
                label="Duplicates Removed"
                value={result.metrics.duplicate_sentences_removed}
              />
            </div>

            <div className="result-grid">
              <div className="panel">
                <div className="panel-title">
                  Optimized Prompt{" "}
                  <span>{result.metrics.optimized_tokens} tokens</span>
                </div>
                <pre>{result.assembled_prompt}</pre>
              </div>

              <div className="panel">
                <div className="panel-title">Compression Pipeline</div>
                <div className="stage-list">
                  {Object.entries(result.metrics.stage_tokens).map(
                    ([name, value]) => (
                      <div className="stage" key={name}>
                        <span>{name.replaceAll("_", " ")}</span>
                        <b>{value.toLocaleString()}</b>
                      </div>
                    )
                  )}
                </div>
                <div className="protection-card">
                  <CheckCircle2 size={18} />
                  <div>
                    <b>Negation-safe optimization</b>
                    <p>
                      Critical words such as not, never, must, should, without
                      and cannot are preserved.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        <section className="feature-grid">
          <Feature
            icon={<FileType />}
            title="Smart Extraction"
            text="PDF parsing with typed + scanned page detection and clean normalization."
          />
          <Feature
            icon={<ImageIcon />}
            title="Intelligent Image Routing"
            text="Automatically classifies handwritten, printed, mixed, photo and blank images."
          />
          <Feature
            icon={<Zap />}
            title="Semantic Optimization"
            text="Phrase compaction, negation protection, deduplication and importance-aware selection."
          />
          <Feature
            icon={<BarChart3 />}
            title="Transparent Metrics"
            text="See exactly how token counts change at every stage of the pipeline."
          />
        </section>
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

function Feature({ icon, title, text }) {
  return (
    <div className="feature">
      <div className="feature-icon">{icon}</div>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

export default App;