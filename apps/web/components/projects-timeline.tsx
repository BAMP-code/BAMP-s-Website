"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { projects } from "@/content/projects";

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

function InlineVideo({ src, alt }: { src: string; alt: string }) {
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handlePlay = useCallback(() => {
    setPlaying(true);
  }, []);

  useEffect(() => {
    if (playing && videoRef.current) {
      videoRef.current.play();
    }
  }, [playing]);

  if (playing) {
    return (
      <video
        ref={videoRef}
        muted
        loop
        playsInline
        preload="metadata"
        className="h-full w-full object-contain"
        aria-label={alt}
      >
        <source src={src} type="video/mp4" />
      </video>
    );
  }

  return (
    <button
      onClick={handlePlay}
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

export function ProjectsTimeline() {
  const sectionRef = useRef<HTMLElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const translateXRef = useRef(0);
  const maxTranslateRef = useRef(0);

  const [isCompact, setIsCompact] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onMotionChange = () => setReduceMotion(query.matches);
    onMotionChange();
    query.addEventListener("change", onMotionChange);
    return () => query.removeEventListener("change", onMotionChange);
  }, []);

  // Calculate max translate on mount/resize
  useEffect(() => {
    const recalc = () => {
      const track = trackRef.current;
      if (!track) return;

      const compact = window.innerWidth < MOBILE_BREAKPOINT;
      setIsCompact(compact);

      if (compact || reduceMotion) {
        maxTranslateRef.current = 0;
        return;
      }

      maxTranslateRef.current = Math.max(0, track.scrollWidth - window.innerWidth + 160);
    };

    recalc();
    window.addEventListener("resize", recalc);
    return () => window.removeEventListener("resize", recalc);
  }, [reduceMotion]);

  // Wheel-to-scroll: only when cursor is over the scroll area
  useEffect(() => {
    if (isCompact || reduceMotion) return;

    const scrollArea = scrollAreaRef.current;
    const track = trackRef.current;
    if (!scrollArea || !track) return;

    const onWheel = (e: WheelEvent) => {
      const max = maxTranslateRef.current;
      if (max <= 0) return;

      const delta = e.deltaY;
      const prev = translateXRef.current;

      // At the edges, let the page scroll through
      if (delta > 0 && prev >= max) return;
      if (delta < 0 && prev <= 0) return;

      e.preventDefault();

      const next = Math.max(0, Math.min(max, prev + delta));
      translateXRef.current = next;
      track.style.transform = `translate3d(${-next}px, 0, 0)`;

      const progress = max > 0 ? next / max : 0;
      const nextActive = Math.round(progress * (projects.length - 1));
      setActiveIndex((p) => (p === nextActive ? p : nextActive));
    };

    scrollArea.addEventListener("wheel", onWheel, { passive: false });
    return () => scrollArea.removeEventListener("wheel", onWheel);
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
        <p className="mb-2 text-xs uppercase tracking-[0.28em] text-white/45">Selected Work</p>
        <h2 className="text-3xl font-semibold text-white sm:text-5xl">Projects Timeline</h2>
        <p className="mt-3 w-[min(92vw,980px)] text-sm leading-relaxed text-white/65 sm:text-base">
          {isCompact || reduceMotion
            ? "Browse through my projects below."
            : "Hover over the timeline and scroll to explore projects."}
        </p>
      </div>

      {isCompact || reduceMotion ? (
        <div className="space-y-10 pb-8">
          {timelineProjects.map((project, index) => {
            const date = PROJECT_DATES[project.id] ?? "TBD";
            return (
              <article key={project.id} className="space-y-3">
                <p className="text-xs uppercase tracking-[0.2em] text-white/45">{date}</p>
                <h3 className="text-xl font-semibold text-white">{project.title}</h3>
                <div className="overflow-hidden rounded-2xl">
                  <div className="h-[220px]">
                    {project.media.type === "video" ? (
                      <InlineVideo
                        src={project.media.src}
                        alt={project.media.alt}
                      />
                    ) : (
                      <Image
                        src={project.media.src}
                        alt={project.media.alt}
                        width={project.media.width}
                        height={project.media.height}
                        className="h-full w-full object-contain"
                        loading="eager"
                        sizes="92vw"
                        quality={75}
                      />
                    )}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <>
          <div
            ref={scrollAreaRef}
            className="relative cursor-ew-resize overflow-hidden py-10"
          >
            <div
              ref={trackRef}
              className="flex items-center gap-12 pb-2 pr-[20vw] will-change-transform"
              style={{ transform: "translate3d(0px, 0, 0)" }}
            >
              {timelineProjects.map((project, index) => {
                const date = PROJECT_DATES[project.id] ?? "TBD";
                const widthClass = CARD_WIDTHS[index % CARD_WIDTHS.length];
                const mediaHeightClass = MEDIA_HEIGHTS[index % MEDIA_HEIGHTS.length];
                const yOffset = index % 2 === 0 ? -38 : 38;
                return (
                <article
                  key={project.id}
                  className={`${widthClass} shrink-0`}
                  style={{ transform: `translateY(${yOffset}px)` }}
                >
                  <p className="mb-2 text-xs uppercase tracking-[0.22em] text-white/45">{date}</p>
                  <h3 className="mb-4 text-2xl font-semibold text-white">{project.title}</h3>
                  <div className={`overflow-hidden rounded-3xl ${mediaHeightClass}`}>
                    {project.media.type === "video" ? (
                      <InlineVideo
                        src={project.media.src}
                        alt={project.media.alt}
                      />
                    ) : (
                      <Image
                        src={project.media.src}
                        alt={project.media.alt}
                        width={project.media.width}
                        height={project.media.height}
                        className="h-full w-full object-contain"
                        loading="eager"
                        sizes="(max-width: 900px) 92vw, 42vw"
                        quality={75}
                      />
                    )}
                  </div>
                </article>
                );
              })}
            </div>
          </div>

          <div className="mt-4 h-[2px] w-full rounded-full bg-white/12">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#ff5138] to-[#ffd066] transition-[width] duration-150"
              style={{ width: `${Math.max(8, ((activeIndex + 1) / timelineProjects.length) * 100)}%` }}
            />
          </div>
        </>
      )}
    </section>
  );
}
