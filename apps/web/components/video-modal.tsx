"use client";

import { useEffect, useRef, useState, useCallback } from "react";

function formatTime(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function VideoModal({
  src,
  alt,
  onClose,
}: {
  src: string;
  alt: string;
  onClose: () => void;
}) {
  const backdropRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const scrubberRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef(0);
  const progressRef = useRef<HTMLDivElement>(null);
  const timeRef = useRef<HTMLSpanElement>(null);
  const [paused, setPaused] = useState(false);

  // Escape to close + lock body scroll
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === " ") {
        e.preventDefault();
        const v = videoRef.current;
        if (!v) return;
        if (v.paused) { v.play(); setPaused(false); }
        else { v.pause(); setPaused(true); }
      }
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  // Auto-play + RAF progress bar (no React re-renders)
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.play();

    let running = true;
    const tick = () => {
      if (!running) return;
      if (v.duration && progressRef.current && timeRef.current) {
        const pct = (v.currentTime / v.duration) * 100;
        progressRef.current.style.width = `${pct}%`;
        timeRef.current.textContent = `${formatTime(v.currentTime)} / ${formatTime(v.duration)}`;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => { running = false; cancelAnimationFrame(rafRef.current); };
  }, []);

  // Scrubber seek
  const seek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const v = videoRef.current;
    const bar = scrubberRef.current;
    if (!v || !bar || !v.duration) return;
    const rect = bar.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    v.currentTime = ratio * v.duration;
  }, []);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPaused(false); }
    else { v.pause(); setPaused(true); }
  }, []);

  return (
    <div // eslint-disable-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions
      ref={backdropRef}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90"
      onClick={(e) => { if (e.target === backdropRef.current) onClose(); }}
      role="dialog"
      aria-label={alt}
    >
      <div className="relative w-[min(92vw,480px)] rounded-2xl border border-white/10 bg-surface-alt shadow-card overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/8">
          <span className="text-xs uppercase tracking-[0.2em] text-muted">{alt}</span>
          <button
            onClick={onClose}
            aria-label="Close video"
            className="flex h-7 w-7 items-center justify-center rounded-full border border-white/12 text-muted transition-colors hover:border-white/30 hover:text-white"
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path d="M1 1l8 8M9 1l-8 8" />
            </svg>
          </button>
        </div>

        {/* Video */}
        <video
          ref={videoRef}
          muted
          loop
          playsInline
          preload="metadata"
          className="h-auto max-h-[72vh] w-full"
          aria-label={alt}
          onClick={togglePlay}
        >
          <source src={src} type="video/mp4" />
        </video>

        {/* Controls */}
        <div className="flex items-center gap-3 px-4 py-3 border-t border-white/8">
          {/* Play/Pause */}
          <button
            onClick={togglePlay}
            aria-label={paused ? "Play" : "Pause"}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-accent/25 text-accent/80 transition-colors hover:border-accent hover:text-accent"
          >
            {paused ? (
              <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor" aria-hidden="true">
                <path d="M0 0l10 6-10 6V0z" />
              </svg>
            ) : (
              <svg width="8" height="10" viewBox="0 0 8 10" fill="currentColor" aria-hidden="true">
                <rect x="0" y="0" width="2.5" height="10" rx="0.5" />
                <rect x="5.5" y="0" width="2.5" height="10" rx="0.5" />
              </svg>
            )}
          </button>

          {/* Scrubber */}
          <div // eslint-disable-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
            ref={scrubberRef}
            className="relative flex-1 h-1.5 cursor-pointer rounded-full bg-white/10"
            onClick={seek}
          >
            <div
              ref={progressRef}
              className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-accent-secondary to-brand-core"
              style={{ width: "0%" }}
            />
          </div>

          {/* Time */}
          <span ref={timeRef} className="shrink-0 text-[11px] tabular-nums text-muted">
            0:00 / 0:00
          </span>
        </div>
      </div>
    </div>
  );
}
