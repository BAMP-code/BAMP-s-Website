"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { projects } from "@/content/projects";
import type { Project } from "@/lib/types";

const MOBILE_BREAKPOINT = 900;
const CARD_WIDTHS = [
  "w-[clamp(300px,34vw,640px)]",
  "w-[clamp(340px,42vw,760px)]",
  "w-[clamp(320px,37vw,690px)]",
  "w-[clamp(360px,46vw,820px)]",
  "w-[clamp(300px,35vw,650px)]",
];
const MEDIA_HEIGHTS = ["h-[220px]", "h-[300px]", "h-[250px]", "h-[340px]", "h-[270px]"];
const PROJECT_DATES: Record<string, string> = {
  "waypoint-prediction": "2025",
  "bio-printing-ui": "2024",
  "website-for-her": "2023",
  "link-app": "2026",
  ecg: "2022",
  truss: "2021",
  "led-board": "2022",
  "useless-box": "2023",
  "cook-drawing": "2020",
  "unnamed-drawing": "2021",
};

function formatTime(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function VideoModal({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
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

function VideoThumbnail({ src, alt, onPlay }: { src: string; alt: string; onPlay: () => void }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onPlay(); }}
      aria-label={`Play video: ${alt}`}
      className="group relative flex h-full w-full items-center justify-center overflow-hidden border-0 bg-[radial-gradient(circle_at_20%_20%,#2a2a33_0%,#14141b_52%,#09090d_100%)]"
    >
      <span className="absolute inset-0 bg-[linear-gradient(130deg,transparent_0%,rgba(255,255,255,0.04)_45%,transparent_100%)]" />
      <span className="relative flex h-14 w-14 items-center justify-center rounded-full border border-white/35 bg-black/45 text-white/90 transition-transform duration-300 group-hover:scale-110">
        <svg width="20" height="22" viewBox="0 0 20 22" fill="currentColor" aria-hidden="true">
          <path d="M2 1.5v19l16-9.5L2 1.5z" />
        </svg>
      </span>
    </button>
  );
}

function ProjectDetails({ project, open }: { project: Project; open: boolean }) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(0);

  useEffect(() => {
    if (!contentRef.current) return;
    setHeight(open ? contentRef.current.scrollHeight : 0);
  }, [open]);

  return (
    <div
      className="overflow-hidden transition-[height] duration-300 ease-in-out"
      style={{ height }}
    >
      <div ref={contentRef} className="pt-3 pb-1">
        <p className="text-sm leading-relaxed text-muted">{project.description}</p>
        {project.links && project.links.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {project.links.map((link) => (
              <a
                key={link.url}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-accent/25 px-3 py-1 text-xs text-accent/80 transition-colors hover:border-accent hover:text-accent"
              >
                {link.label} &rarr;
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function ProjectsTimeline() {
  const sectionRef = useRef<HTMLElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);

  const [isCompact, setIsCompact] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [videoModal, setVideoModal] = useState<{ src: string; alt: string } | null>(null);

  const toggleExpand = useCallback((id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  }, []);

  const closeVideo = useCallback(() => setVideoModal(null), []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onMotionChange = () => setReduceMotion(query.matches);
    onMotionChange();
    query.addEventListener("change", onMotionChange);
    return () => query.removeEventListener("change", onMotionChange);
  }, []);

  useEffect(() => {
    const recalc = () => {
      setIsCompact(window.innerWidth < MOBILE_BREAKPOINT);
    };
    recalc();
    window.addEventListener("resize", recalc);
    return () => window.removeEventListener("resize", recalc);
  }, []);

  // Update progress bar from native horizontal scroll position.
  // No wheel hijacking — users get OS-native horizontal scroll
  // (trackpad swipe, shift+wheel, touch swipe, scrollbar drag).
  useEffect(() => {
    if (isCompact || reduceMotion) return;

    const scrollArea = scrollAreaRef.current;
    if (!scrollArea) return;

    const updateBar = () => {
      const bar = progressBarRef.current;
      if (!bar) return;
      const max = scrollArea.scrollWidth - scrollArea.clientWidth;
      const progress = max > 0 ? scrollArea.scrollLeft / max : 0;
      bar.style.width = `${Math.max(8, progress * 100)}%`;
    };

    updateBar();
    scrollArea.addEventListener("scroll", updateBar, { passive: true });
    window.addEventListener("resize", updateBar);
    return () => {
      scrollArea.removeEventListener("scroll", updateBar);
      window.removeEventListener("resize", updateBar);
    };
  }, [isCompact, reduceMotion]);

  const timelineProjects = useMemo(() => {
    return [...projects].sort((a, b) => {
      const aYear = Number(PROJECT_DATES[a.id] ?? "9999");
      const bYear = Number(PROJECT_DATES[b.id] ?? "9999");
      return aYear - bYear;
    });
  }, []);

  return (
    <section
      id="projects"
      ref={sectionRef}
      className="scroll-mt-32 bg-black px-[clamp(16px,3.5vw,72px)] py-16"
      aria-label="Projects timeline"
    >
      <div className="mb-8">
        <p className="mb-2 text-xs uppercase tracking-[0.28em] text-muted">Selected Work</p>
        <h2 className="text-3xl font-semibold text-white sm:text-5xl">Projects Timeline</h2>
        <p className="mt-3 w-[min(92vw,980px)] text-sm leading-relaxed text-muted sm:text-base">
          {isCompact || reduceMotion
            ? "Browse through my projects below."
            : "Swipe or shift-scroll to explore the timeline."}
        </p>
      </div>

      {isCompact || reduceMotion ? (
        <div className="space-y-10 pb-8">
          {timelineProjects.map((project, index) => {
            const date = PROJECT_DATES[project.id] ?? "TBD";
            const isOpen = expandedId === project.id;
            return (
              <div
                key={project.id}
                className="cursor-pointer space-y-3"
                onClick={() => toggleExpand(project.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleExpand(project.id); } }}
                aria-expanded={isOpen}
              >
                <p className="text-xs uppercase tracking-[0.2em] text-muted">{date}</p>
                <h3 className="text-xl font-semibold text-white">{project.title}</h3>
                <div className="overflow-hidden rounded-2xl">
                  <div className="h-[220px]">
                    {project.media.type === "video" ? (
                      <VideoThumbnail
                        src={project.media.src}
                        alt={project.media.alt}
                        onPlay={() => setVideoModal({ src: project.media.src, alt: project.media.alt })}
                      />
                    ) : (
                      <Image
                        src={project.media.src}
                        alt={project.media.alt}
                        width={project.media.width}
                        height={project.media.height}
                        className="h-full w-full object-contain"
                        loading={index < 3 ? "eager" : "lazy"}
                        priority={index < 2}
                        sizes="92vw"
                        quality={75}
                      />
                    )}
                  </div>
                </div>
                <ProjectDetails project={project} open={isOpen} />
              </div>
            );
          })}
        </div>
      ) : (
        <>
          <div
            ref={scrollAreaRef}
            className="relative overflow-x-auto overflow-y-hidden snap-x snap-mandatory py-10 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
          >
            <div
              ref={trackRef}
              className="flex items-center gap-12 pb-2 pr-[20vw]"
            >
              {timelineProjects.map((project, index) => {
                const date = PROJECT_DATES[project.id] ?? "TBD";
                const widthClass = CARD_WIDTHS[index % CARD_WIDTHS.length];
                const mediaHeightClass = MEDIA_HEIGHTS[index % MEDIA_HEIGHTS.length];
                const yOffset = index % 2 === 0 ? -38 : 38;
                const isOpen = expandedId === project.id;
                return (
                <div
                  key={project.id}
                  className={`${widthClass} shrink-0 cursor-pointer snap-start`}
                  style={{ transform: `translateY(${yOffset}px)` }}
                  onClick={() => toggleExpand(project.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleExpand(project.id); } }}
                  aria-expanded={isOpen}
                >
                  <p className="mb-2 text-xs uppercase tracking-[0.22em] text-muted">{date}</p>
                  <h3 className="mb-4 text-2xl font-semibold text-white">{project.title}</h3>
                  <div className={`overflow-hidden rounded-3xl transition-shadow duration-200 hover:shadow-[0_0_0_1px_rgba(0,246,255,0.15)] ${mediaHeightClass}`}>
                    {project.media.type === "video" ? (
                      <VideoThumbnail
                        src={project.media.src}
                        alt={project.media.alt}
                        onPlay={() => setVideoModal({ src: project.media.src, alt: project.media.alt })}
                      />
                    ) : (
                      <Image
                        src={project.media.src}
                        alt={project.media.alt}
                        width={project.media.width}
                        height={project.media.height}
                        className="h-full w-full object-contain"
                        loading={index < 3 ? "eager" : "lazy"}
                        priority={index < 2}
                        sizes="(max-width: 900px) 92vw, 42vw"
                        quality={75}
                      />
                    )}
                  </div>
                  <ProjectDetails project={project} open={isOpen} />
                </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 h-[2px] w-full rounded-full bg-border">
            <div
              ref={progressBarRef}
              className="h-full rounded-full bg-gradient-to-r from-accent-secondary to-brand-core"
              style={{ width: "8%", transition: "none" }}
            />
          </div>
        </>
      )}
      {videoModal && (
        <VideoModal src={videoModal.src} alt={videoModal.alt} onClose={closeVideo} />
      )}
    </section>
  );
}
