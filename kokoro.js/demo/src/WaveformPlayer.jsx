import { useRef, useEffect, useState, useCallback } from "react";

const BAR_COUNT = 60;
const BAR_GAP = 2;

function formatTime(secs) {
  if (!isFinite(secs)) return "0:00";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function WaveformPlayer({ src }) {
  const canvasRef = useRef(null);
  const audioRef = useRef(null);
  const animRef = useRef(null);
  const waveformRef = useRef(null); // sampled amplitude data

  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [ready, setReady] = useState(false);

  // Decode audio and sample waveform amplitudes
  useEffect(() => {
    if (!src) return;
    let cancelled = false;

    (async () => {
      try {
        const ctx = new AudioContext();
        const response = await fetch(src);
        const arrayBuf = await response.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(arrayBuf);
        if (cancelled) { ctx.close(); return; }

        const raw = audioBuf.getChannelData(0);
        const blockSize = Math.floor(raw.length / BAR_COUNT);
        const bars = new Float32Array(BAR_COUNT);
        for (let i = 0; i < BAR_COUNT; i++) {
          let sum = 0;
          for (let j = 0; j < blockSize; j++) sum += Math.abs(raw[i * blockSize + j]);
          bars[i] = sum / blockSize;
        }
        // Normalize
        const max = Math.max(...bars, 0.001);
        for (let i = 0; i < BAR_COUNT; i++) bars[i] /= max;

        waveformRef.current = bars;
        setDuration(audioBuf.duration);
        setReady(true);
        ctx.close();
      } catch (e) {
        // Fallback: random-ish waveform shape
        if (cancelled) return;
        const bars = new Float32Array(BAR_COUNT);
        for (let i = 0; i < BAR_COUNT; i++) {
          bars[i] = 0.2 + Math.random() * 0.6;
        }
        waveformRef.current = bars;
        setReady(true);
      }
    })();

    return () => { cancelled = true; };
  }, [src]);

  // Draw waveform
  const drawWaveform = useCallback((progress = 0) => {
    const canvas = canvasRef.current;
    const bars = waveformRef.current;
    if (!canvas || !bars) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const barW = (W - BAR_GAP * (BAR_COUNT - 1)) / BAR_COUNT;
    const progressIdx = progress * BAR_COUNT;

    for (let i = 0; i < BAR_COUNT; i++) {
      const x = i * (barW + BAR_GAP);
      const amp = bars[i];
      const barH = Math.max(2, amp * (H - 4));
      const y = (H - barH) / 2;

      const played = i < progressIdx;
      const cursor = Math.abs(i - progressIdx) < 1;

      ctx.fillStyle = cursor
        ? "rgba(232,160,48,1)"
        : played
        ? "rgba(232,160,48,0.7)"
        : "rgba(30,38,54,1)";

      // Rounded bars
      const r = Math.min(barW / 2, 2);
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, r);
      ctx.fill();
    }
  }, []);

  // Draw on resize / waveform ready
  useEffect(() => {
    if (!ready) return;
    drawWaveform(currentTime / (duration || 1));

    const ro = new ResizeObserver(() => drawWaveform(currentTime / (duration || 1)));
    if (canvasRef.current) ro.observe(canvasRef.current);
    return () => ro.disconnect();
  }, [ready, drawWaveform, currentTime, duration]);

  // Animation loop for playhead
  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(animRef.current);
      return;
    }
    const tick = () => {
      const audio = audioRef.current;
      if (audio) {
        const t = audio.currentTime;
        const d = audio.duration || 1;
        setCurrentTime(t);
        drawWaveform(t / d);
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing, drawWaveform]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      audio.pause();
    } else {
      audio.play().catch(() => {});
    }
  };

  const handleSeek = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = x / rect.width;
    const audio = audioRef.current;
    if (audio && isFinite(duration)) {
      audio.currentTime = pct * duration;
      setCurrentTime(pct * duration);
      drawWaveform(pct);
    }
  };

  return (
    <div className="flex items-center gap-3 w-full">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={src}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => { setPlaying(false); setCurrentTime(0); drawWaveform(0); }}
        onLoadedMetadata={(e) => setDuration(e.target.duration)}
      />

      {/* Play/Pause button */}
      <button
        type="button"
        onClick={togglePlay}
        className="flex-shrink-0 flex items-center justify-center rounded-full transition-all"
        style={{
          width: 36,
          height: 36,
          background: playing ? "var(--accent)" : "var(--bg2)",
          border: "1px solid var(--border-lit)",
          color: playing ? "#000" : "var(--text1)",
        }}
        onMouseEnter={(e) => { if (!playing) e.currentTarget.style.borderColor = "var(--accent-dim)"; }}
        onMouseLeave={(e) => { if (!playing) e.currentTarget.style.borderColor = "var(--border-lit)"; }}
      >
        {playing ? (
          // Pause icon
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
            <rect x="2" y="1.5" width="3" height="9" rx="1" />
            <rect x="7" y="1.5" width="3" height="9" rx="1" />
          </svg>
        ) : (
          // Play icon
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
            <path d="M3 1.5l7 4.5-7 4.5V1.5z" />
          </svg>
        )}
      </button>

      {/* Waveform canvas */}
      <div className="flex-1 relative" style={{ height: 40, cursor: "pointer" }} onClick={handleSeek}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center gap-0.5">
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className="rounded-sm"
                style={{
                  width: 3,
                  height: `${20 + Math.sin(i * 0.7) * 12}%`,
                  background: "var(--border-lit)",
                  animation: `bar-idle ${0.8 + (i % 4) * 0.15}s ease-in-out ${i * 0.05}s infinite`,
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Time */}
      <span
        className="flex-shrink-0 text-xs tabular-nums"
        style={{ color: "var(--text3)", fontFamily: "Space Mono, monospace", minWidth: 72, textAlign: "right" }}
      >
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>

      {/* Download */}
      <a
        href={src}
        download="kokoro-speech.wav"
        className="flex-shrink-0 flex items-center justify-center rounded transition-colors"
        style={{ width: 28, height: 28, color: "var(--text3)" }}
        title="Download audio"
        onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text1)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text3)"; }}
      >
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <path d="M6.5 1v7.5M3.5 6l3 3 3-3M1.5 10.5v1a1 1 0 001 1h9a1 1 0 001-1v-1" />
        </svg>
      </a>
    </div>
  );
}
