import { useRef, useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import WaveformPlayer from "./WaveformPlayer.jsx";

// ── language metadata ──────────────────────────────────────────────────────────
const LANG_META = {
  "en-us": { label: "American English", flag: "🇺🇸", short: "EN-US" },
  "en-gb": { label: "British English",  flag: "🇬🇧", short: "EN-GB" },
  ja:      { label: "Japanese",         flag: "🇯🇵", short: "JA" },
  zh:      { label: "Chinese",          flag: "🇨🇳", short: "ZH" },
  es:      { label: "Spanish",          flag: "🇪🇸", short: "ES" },
  fr:      { label: "French",           flag: "🇫🇷", short: "FR" },
  hi:      { label: "Hindi",            flag: "🇮🇳", short: "HI" },
  it:      { label: "Italian",          flag: "🇮🇹", short: "IT" },
  "pt-br": { label: "Portuguese (BR)",  flag: "🇧🇷", short: "PT" },
};

const GRADE_COLOR = {
  A: "#39d98a", "A-": "#39d98a",
  B: "#e8a030", "B-": "#e8a030", "B+": "#e8a030",
  C: "#7a8ca0", "C+": "#7a8ca0", "C-": "#7a8ca0",
  D: "#3a4a5a", "D+": "#3a4a5a", "D-": "#3a4a5a",
  F: "#3a4a5a", "F+": "#3a4a5a",
};

function groupVoicesByLanguage(voices) {
  const groups = {};
  for (const [id, v] of Object.entries(voices)) {
    const lang = v.language;
    if (!groups[lang]) groups[lang] = [];
    groups[lang].push({ id, ...v });
  }
  return groups;
}

// ── VoiceSelector ──────────────────────────────────────────────────────────────
function VoiceSelector({ voices, selected, onSelect }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const groups = groupVoicesByLanguage(voices);
  const selectedVoice = voices[selected];

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left transition-colors"
        style={{ background: "var(--bg1)", border: "1px solid var(--border-lit)", borderRadius: 6 }}
      >
        <span className="text-xl">{LANG_META[selectedVoice?.language]?.flag ?? "🌐"}</span>
        <span className="flex-1 min-w-0">
          <span className="block text-sm font-bold" style={{ color: "var(--text1)", fontFamily: "Syne" }}>
            {selectedVoice?.name ?? selected}
            {selectedVoice?.traits && <span className="ml-1">{selectedVoice.traits}</span>}
          </span>
          <span className="block text-xs mt-0.5" style={{ color: "var(--text2)", fontFamily: "Space Mono, monospace" }}>
            {LANG_META[selectedVoice?.language]?.label} · {selectedVoice?.gender}
          </span>
        </span>
        <span
          className="text-xs px-1.5 py-0.5 font-bold"
          style={{
            color: GRADE_COLOR[selectedVoice?.overallGrade] ?? "var(--text3)",
            border: `1px solid ${GRADE_COLOR[selectedVoice?.overallGrade] ?? "var(--text3)"}`,
            borderRadius: 3,
            fontFamily: "Space Mono, monospace",
          }}
        >
          {selectedVoice?.overallGrade}
        </span>
        <svg
          width="12" height="12" viewBox="0 0 12 12" fill="none"
          style={{ color: "var(--text2)", transform: open ? "rotate(180deg)" : "none", transition: "transform 0.2s", flexShrink: 0 }}
        >
          <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 w-full mt-1 overflow-auto"
            style={{
              background: "var(--bg1)",
              border: "1px solid var(--border-lit)",
              borderRadius: 6,
              maxHeight: 320,
              boxShadow: "0 16px 40px rgba(0,0,0,0.6)",
            }}
          >
            {Object.entries(groups).map(([lang, voiceList]) => {
              const meta = LANG_META[lang] ?? { label: lang, flag: "🌐", short: lang };
              return (
                <div key={lang}>
                  <div
                    className="px-3 py-1.5 text-xs font-bold sticky top-0 flex items-center gap-2"
                    style={{ background: "var(--bg)", color: "var(--text2)", fontFamily: "Space Mono, monospace", borderBottom: "1px solid var(--border)" }}
                  >
                    <span>{meta.flag}</span>
                    <span>{meta.label}</span>
                  </div>
                  {voiceList.map((v) => (
                    <button
                      key={v.id}
                      type="button"
                      onClick={() => { onSelect(v.id); setOpen(false); }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors"
                      style={{
                        background: v.id === selected ? "var(--accent-glow)" : "transparent",
                        borderLeft: v.id === selected ? "2px solid var(--accent)" : "2px solid transparent",
                      }}
                      onMouseEnter={(e) => { if (v.id !== selected) e.currentTarget.style.background = "var(--bg2)"; }}
                      onMouseLeave={(e) => { if (v.id !== selected) e.currentTarget.style.background = "transparent"; }}
                    >
                      <span className="flex-1 min-w-0">
                        <span className="block text-sm" style={{ color: v.id === selected ? "var(--accent)" : "var(--text1)", fontFamily: "Syne" }}>
                          {v.name}
                          {v.traits && <span className="ml-1 text-xs">{v.traits}</span>}
                        </span>
                        <span className="block text-xs" style={{ color: "var(--text3)", fontFamily: "Space Mono, monospace" }}>
                          {v.gender} · {v.id}
                        </span>
                      </span>
                      <span
                        className="text-xs px-1 py-0.5"
                        style={{
                          color: GRADE_COLOR[v.overallGrade] ?? "var(--text3)",
                          fontFamily: "Space Mono, monospace",
                          flexShrink: 0,
                        }}
                      >
                        {v.overallGrade}
                      </span>
                    </button>
                  ))}
                </div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── LoadingScreen ──────────────────────────────────────────────────────────────
function LoadingScreen({ message, error, progress }) {
  const percent = progress?.progress ?? 0;
  const loaded = progress?.loaded ?? 0;
  const total = progress?.total ?? 0;
  const mbLoaded = (loaded / 1024 / 1024).toFixed(1);
  const mbTotal = (total / 1024 / 1024).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6 }}
      className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-8"
      style={{ background: "var(--bg)" }}
    >
      {/* Kanji mark */}
      <div className="relative flex items-center justify-center" style={{ width: 120, height: 120 }}>
        {/* Ring */}
        {!error && (
          <svg width="120" height="120" viewBox="0 0 120 120" style={{ position: "absolute", top: 0, left: 0, transform: "rotate(-90deg)" }}>
            <circle cx="60" cy="60" r="54" fill="none" stroke="var(--border)" strokeWidth="2" />
            <circle
              cx="60" cy="60" r="54" fill="none"
              stroke="var(--accent)" strokeWidth="2"
              strokeDasharray={`${2 * Math.PI * 54}`}
              strokeDashoffset={`${2 * Math.PI * 54 * (1 - percent / 100)}`}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 0.3s ease" }}
            />
          </svg>
        )}
      </div>

      <div className="text-center space-y-2">
        <p
          className="text-lg font-bold"
          style={{ color: error ? "var(--red)" : "var(--text1)", fontFamily: "Syne" }}
        >
          {error ?? "Loading model"}
        </p>
        {!error && (
          <p className="text-sm" style={{ color: "var(--text2)", fontFamily: "Space Mono, monospace" }}>
            {progress?.file
              ? `${progress.file.split("/").pop()} · ${mbLoaded}/${mbTotal} MB`
              : message}
          </p>
        )}
        {!error && percent > 0 && (
          <p className="text-xs" style={{ color: "var(--text3)", fontFamily: "Space Mono, monospace" }}>
            {percent.toFixed(0)}%
          </p>
        )}
      </div>
    </motion.div>
  );
}

// ── ResultCard ─────────────────────────────────────────────────────────────────
function ResultCard({ result, index, voices }) {
  const voice = voices[result.voice];
  const lang = voice?.language;
  const meta = LANG_META[lang] ?? { flag: "🌐" };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="rounded-lg overflow-hidden"
      style={{ border: "1px solid var(--border-lit)", background: "var(--bg1)" }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <span className="text-lg">{meta.flag}</span>
        <div className="flex-1 min-w-0">
          <span className="text-xs font-bold" style={{ color: "var(--accent)", fontFamily: "Space Mono, monospace" }}>
            #{String(index + 1).padStart(2, "0")} · {voice?.name ?? result.voice}
            {voice?.traits && <span className="ml-1">{voice.traits}</span>}
          </span>
        </div>
        {result.isStreaming && (
          <span className="flex items-center gap-1.5 text-xs" style={{ color: "var(--teal)", fontFamily: "Space Mono, monospace" }}>
            <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: "var(--teal)", animation: "pulse-dot 1s ease infinite" }} />
            generating
          </span>
        )}
      </div>

      {/* Chunks */}
      <div className="divide-y" style={{ borderColor: "var(--border)" }}>
        {result.chunks.map((chunk, ci) => (
          <div key={ci} className="px-4 py-3 space-y-3">
            <p className="text-sm leading-relaxed" style={{ color: "var(--text1)", fontFamily: "Space Mono, monospace" }}>
              {chunk.text}
            </p>
            <WaveformPlayer src={chunk.audio} />
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────
const EXAMPLE_TEXTS = {
  "en-us": "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, and the shells she sells are surely seashells.",
  "en-gb": "Shall I compare thee to a summer's day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May.",
  // Japanese: use hiragana/katakana — kanji is not supported by the browser phonemizer
  ja: "こんにちは。おはよう ございます。ありがとう ございます。またね。",
  zh: "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
  es: "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo.",
  fr: "Il était une fois, dans un pays très lointain, une belle princesse qui vivait dans un grand château.",
  hi: "आज का मौसम बहुत अच्छा है। हम सब मिलकर काम करते हैं।",
  it: "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura.",
  "pt-br": "No meio do caminho tinha uma pedra, tinha uma pedra no meio do caminho.",
};

export default function App() {
  const worker = useRef(null);

  const [status, setStatus] = useState(null); // null=loading, ready, running
  const [device, setDevice] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState("Initializing…");
  const [loadingProgress, setLoadingProgress] = useState(null);
  const [error, setError] = useState(null);

  const [voices, setVoices] = useState({});
  const [selected, setSelected] = useState("af_heart");
  const [inputText, setInputText] = useState(EXAMPLE_TEXTS["en-us"]);
  const [speed, setSpeed] = useState(1);
  const [results, setResults] = useState([]);

  // Setup worker
  useEffect(() => {
    worker.current ??= new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

    const onMessage = (e) => {
      const { status: s, ...data } = e.data;
      switch (s) {
        case "device":
          setDevice(data.device);
          setLoadingMessage(`Loading model on ${data.device.toUpperCase()}…`);
          break;
        case "progress":
          setLoadingProgress(data.progress);
          break;
        case "ready":
          setStatus("ready");
          setVoices(data.voices);
          break;
        case "error":
          setError(data.error);
          break;
        case "chunk":
          setResults((prev) => {
            const updated = [...prev];
            const last = updated[0];
            if (last && last.isStreaming) {
              last.chunks = [...last.chunks, { audio: data.audio, text: data.text, phonemes: data.phonemes }];
              return [last, ...updated.slice(1)];
            }
            return updated;
          });
          break;
        case "complete":
          setStatus("ready");
          setResults((prev) => {
            const updated = [...prev];
            if (updated[0]) updated[0].isStreaming = false;
            return updated;
          });
          break;
      }
    };

    const onError = (e) => { console.error(e); setError(e.message); };
    worker.current.addEventListener("message", onMessage);
    worker.current.addEventListener("error", onError);
    return () => {
      worker.current.removeEventListener("message", onMessage);
      worker.current.removeEventListener("error", onError);
    };
  }, []);

  const handleVoiceChange = useCallback((id) => {
    setSelected(id);
    const lang = voices[id]?.language;
    if (lang && EXAMPLE_TEXTS[lang]) setInputText(EXAMPLE_TEXTS[lang]);
  }, [voices]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim() || status !== "ready") return;
    setStatus("running");
    setResults((prev) => [{ voice: selected, chunks: [], isStreaming: true }, ...prev]);
    worker.current.postMessage({ type: "generate", text: inputText.trim(), voice: selected, speed });
  };

  const isLoading = status === null;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--bg)" }}>
      <AnimatePresence>
        {isLoading && <LoadingScreen message={loadingMessage} error={error} progress={loadingProgress} key="loader" />}
      </AnimatePresence>

      {/* Header */}
      <header
        className="flex items-center justify-between px-6 py-4 sticky top-0 z-10"
        style={{ borderBottom: "1px solid var(--border)", background: "rgba(6, 9, 15, 0.85)", backdropFilter: "blur(12px)" }}
      >
        <div className="flex items-center gap-3">
          <div>
            <h1 className="text-lg font-extrabold leading-none" style={{ color: "var(--text1)" }}>Kokoro TTS</h1>
            <p className="text-xs" style={{ color: "var(--text2)", fontFamily: "Space Mono, monospace" }}>
              {device ? `${device.toUpperCase()} · ` : ""}kokoro-js
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {status === "ready" && (
            <span className="flex items-center gap-1.5 text-xs" style={{ color: "var(--teal)", fontFamily: "Space Mono, monospace" }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--teal)" }} />
              ready
            </span>
          )}
          <a
            href="https://github.com/hexgrad/kokoro"
            target="_blank"
            rel="noreferrer"
            className="text-xs px-3 py-1.5 rounded font-semibold transition-colors"
            style={{
              border: "1px solid var(--border-lit)",
              color: "var(--text2)",
              fontFamily: "Space Mono, monospace",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text1)"; e.currentTarget.style.borderColor = "var(--border-lit)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text2)"; }}
          >
            GitHub ↗
          </a>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex-1 flex flex-col lg:flex-row gap-0 relative z-[1]">
        {/* ── Left: Controls ── */}
        <aside
          className="w-full lg:w-[380px] lg:min-w-[380px] flex flex-col gap-5 p-5 lg:overflow-y-auto lg:sticky lg:top-[61px] lg:h-[calc(100vh-61px)]"
          style={{ borderRight: "1px solid var(--border)" }}
        >
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            {/* Text input */}
            <div>
              <label className="block text-xs font-bold mb-2" style={{ color: "var(--text2)", fontFamily: "Space Mono, monospace" }}>
                INPUT TEXT
              </label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to synthesize…"
                rows={7}
                className="w-full text-sm px-4 py-3 rounded-lg leading-relaxed transition-colors"
                style={{
                  background: "var(--bg1)",
                  border: "1px solid var(--border-lit)",
                  color: "var(--text1)",
                  fontFamily: "Space Mono, monospace",
                  fontSize: 12,
                }}
                onFocus={(e) => { e.target.style.borderColor = "var(--accent-dim)"; }}
                onBlur={(e) => { e.target.style.borderColor = "var(--border-lit)"; }}
              />
            </div>

            {/* Voice selector */}
            <div>
              <label className="block text-xs font-bold mb-2" style={{ color: "var(--text2)", fontFamily: "Space Mono, monospace" }}>
                VOICE
              </label>
              {Object.keys(voices).length > 0 ? (
                <VoiceSelector voices={voices} selected={selected} onSelect={handleVoiceChange} />
              ) : (
                <div className="h-12 rounded-lg animate-pulse" style={{ background: "var(--bg1)" }} />
              )}
            </div>

            {/* Language note */}
            {voices[selected]?.language === "ja" && (
              <div className="text-xs px-3 py-2 rounded" style={{ background: "rgba(232,160,48,0.08)", border: "1px solid rgba(232,160,48,0.25)", color: "var(--text2)", fontFamily: "Space Mono, monospace", lineHeight: 1.6 }}>
                Japanese: use hiragana/katakana (e.g. こんにちは). Kanji is not phonemized in the browser.
              </div>
            )}

            {/* Speed slider */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-bold" style={{ color: "var(--text2)", fontFamily: "Space Mono, monospace" }}>
                  SPEED
                </label>
                <span className="text-xs" style={{ color: "var(--accent)", fontFamily: "Space Mono, monospace" }}>
                  {speed.toFixed(1)}×
                </span>
              </div>
              <div className="relative">
                <input
                  type="range" min="0.5" max="2.0" step="0.1"
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                  className="w-full h-1 rounded-full appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, var(--accent) ${((speed - 0.5) / 1.5) * 100}%, var(--border-lit) ${((speed - 0.5) / 1.5) * 100}%)`,
                    accentColor: "var(--accent)",
                  }}
                />
              </div>
            </div>

            {/* Generate button */}
            <button
              type="submit"
              disabled={status !== "ready" || !inputText.trim()}
              className="w-full py-3.5 rounded-lg font-bold text-sm transition-all relative overflow-hidden"
              style={{
                background: status === "running" ? "var(--bg2)" : "var(--accent)",
                color: status === "running" ? "var(--text2)" : "#000",
                fontFamily: "Syne",
                fontSize: 13,
                letterSpacing: "0.08em",
                cursor: status === "ready" && inputText.trim() ? "pointer" : "not-allowed",
                opacity: status !== "ready" || !inputText.trim() ? 0.6 : 1,
              }}
            >
              {status === "running" ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--teal)", animation: "pulse-dot 1s ease infinite" }} />
                  SYNTHESIZING…
                </span>
              ) : (
                "GENERATE SPEECH"
              )}
            </button>
          </form>

          {/* Footer info */}
          <div className="mt-auto pt-4" style={{ borderTop: "1px solid var(--border)" }}>
            <p className="text-xs leading-relaxed" style={{ color: "var(--text3)", fontFamily: "Space Mono, monospace" }}>
              Powered by{" "}
              <a href="https://github.com/hexgrad/kokoro" target="_blank" rel="noreferrer" style={{ color: "var(--text2)" }}>
                Kokoro
              </a>{" "}
              ·{" "}
              <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer" style={{ color: "var(--text2)" }}>
                Transformers.js
              </a>
              <br />
              Model runs entirely in-browser. No data sent to servers.
            </p>
          </div>
        </aside>

        {/* ── Right: Results ── */}
        <main className="flex-1 p-5 space-y-4 overflow-y-auto">
          {results.length === 0 && status === "ready" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="h-full flex flex-col items-center justify-center gap-3 py-24"
            >
              <p className="text-sm" style={{ color: "var(--text3)", fontFamily: "Space Mono, monospace" }}>
                Generate your first audio to get started
              </p>
            </motion.div>
          )}

          {results.map((result, i) => (
            <ResultCard key={i} result={result} index={results.length - 1 - i} voices={voices} />
          ))}
        </main>
      </div>
    </div>
  );
}
