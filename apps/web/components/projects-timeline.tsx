"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { projects } from "@/content/projects";

type TimelineMetrics = {
  maxTranslate: number;
  pinDistance: number;
};

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

function TimelineVideoPlaceholder({ src, alt }: { src: string; alt: string }) {
  return (
    <a
      href={src}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`Open video demo: ${alt}`}
      className="group relative flex h-full w-full items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_20%_20%,#2a2a33_0%,#14141b_52%,#09090d_100%)]"
    >
      <span className="absolute inset-0 bg-[linear-gradient(130deg,transparent_0%,rgba(255,255,255,0.04)_45%,transparent_100%)]" />
      <span className="relative flex h-14 w-14 items-center justify-center rounded-full border border-white/35 bg-black/45 text-white/90 transition-transform duration-300 group-hover:scale-105">
        ▶
      </span>
      <span className="absolute bottom-3 right-3 text-[10px] uppercase tracking-[0.16em] text-white/55">
        Open demo
      </span>
    </a>
  );
}

export function ProjectsTimeline() {
  const sectionRef = useRef<HTMLElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const progressFillRef = useRef<HTMLDivElement>(null);
  const rafScrollRef = useRef<number | null>(null);
  const progressRef = useRef(0);

  const [metrics, setMetrics] = useState<TimelineMetrics>({
    maxTranslate: 0,
    pinDistance: 0,
  });
  const [isCompact, setIsCompact] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onMotionChange = () => setReduceMotion(query.matches);
    onMotionChange();
    query.addEventListener("change", onMotionChange);

    return () => {
      query.removeEventListener("change", onMotionChange);
    };
  }, []);

  useEffect(() => {
    const recalc = () => {
      const section = sectionRef.current;
      const track = trackRef.current;
      if (!section || !track) return;

      const compact = window.innerWidth < MOBILE_BREAKPOINT;
      setIsCompact(compact);

      if (compact || reduceMotion) {
        setMetrics({ maxTranslate: 0, pinDistance: 0 });
        progressRef.current = 0;
        setActiveIndex(0);
        return;
      }

      const maxTranslate = Math.max(0, track.scrollWidth - window.innerWidth + 64);
      const pinDistance = Math.max(window.innerHeight * 0.75, maxTranslate + window.innerHeight * 0.55);
      setMetrics({ maxTranslate, pinDistance });
    };

    recalc();
    window.addEventListener("resize", recalc);
    return () => {
      window.removeEventListener("resize", recalc);
    };
  }, [reduceMotion]);

  useEffect(() => {
    if (isCompact || reduceMotion) return;

    const updateTrackForScroll = () => {
      const section = sectionRef.current;
      const track = trackRef.current;
      const progressFill = progressFillRef.current;
      if (!section || !track || !progressFill || metrics.pinDistance <= 0) return;

      const top = section.offsetTop;
      const y = window.scrollY;
      const raw = (y - top) / metrics.pinDistance;
      const nextProgress = Math.max(0, Math.min(1, raw));
      if (Math.abs(nextProgress - progressRef.current) < 0.002) return;

      progressRef.current = nextProgress;
      const x = -nextProgress * metrics.maxTranslate;
      track.style.transform = `translate3d(${x}px, 0, 0)`;
      progressFill.style.width = `${Math.max(6, nextProgress * 100)}%`;

      const nextActiveIndex = Math.round(nextProgress * (projects.length - 1));
      setActiveIndex((prev) => (prev === nextActiveIndex ? prev : nextActiveIndex));
    };

    const onScroll = () => {
      if (rafScrollRef.current !== null) return;
      rafScrollRef.current = window.requestAnimationFrame(() => {
        rafScrollRef.current = null;
        updateTrackForScroll();
      });
    };

    updateTrackForScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      if (rafScrollRef.current !== null) {
        window.cancelAnimationFrame(rafScrollRef.current);
      }
      window.removeEventListener("scroll", onScroll);
    };
  }, [isCompact, metrics.maxTranslate, metrics.pinDistance, reduceMotion]);

  const sectionHeight = useMemo(() => {
    if (isCompact || reduceMotion || metrics.pinDistance <= 0) return "auto";
    return `calc(100vh + ${metrics.pinDistance}px)`;
  }, [isCompact, metrics.pinDistance, reduceMotion]);

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
      style={{ minHeight: sectionHeight }}
      aria-label="Projects timeline"
    >
      <div className={isCompact || reduceMotion ? "w-full" : "sticky top-0 h-screen"}>
        <div className="flex h-full w-full flex-col justify-center">
          <div className="mb-8">
            <p className="mb-2 text-xs uppercase tracking-[0.28em] text-white/45">Selected Work</p>
            <h2 className="text-3xl font-semibold text-white sm:text-5xl">Projects Timeline</h2>
            <p className="mt-3 w-[min(92vw,980px)] text-sm leading-relaxed text-white/65 sm:text-base">
              Scroll down to move through all projects from left to right.
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
                          <TimelineVideoPlaceholder
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
                            loading={index < 3 ? "eager" : "lazy"}
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
              <div className="relative overflow-hidden py-10">
                <div
                  ref={trackRef}
                  className="flex items-center gap-12 pb-2 pr-8 will-change-transform"
                  style={{ transform: "translate3d(0px, 0, 0)" }}
                >
                  {timelineProjects.map((project, index) => {
                    const date = PROJECT_DATES[project.id] ?? "TBD";
                    const widthClass = CARD_WIDTHS[index % CARD_WIDTHS.length];
                    const mediaHeightClass = MEDIA_HEIGHTS[index % MEDIA_HEIGHTS.length];
                    const yOffset = index % 2 === 0 ? -38 : 38;
                    const shouldRenderMedia = Math.abs(index - activeIndex) <= 2;
                    return (
                    <article
                      key={project.id}
                      className={`${widthClass} shrink-0`}
                      style={{ transform: `translateY(${yOffset}px)` }}
                    >
                      <p className="mb-2 text-xs uppercase tracking-[0.22em] text-white/45">{date}</p>
                      <h3 className="mb-4 text-2xl font-semibold text-white">{project.title}</h3>
                      <div className={`overflow-hidden rounded-3xl ${mediaHeightClass}`}>
                        {shouldRenderMedia ? (
                          project.media.type === "video" ? (
                            <TimelineVideoPlaceholder
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
                              loading={index < 3 ? "eager" : "lazy"}
                              sizes="(max-width: 900px) 92vw, 42vw"
                              quality={75}
                            />
                          )
                        ) : (
                          <div className="flex h-full w-full items-center justify-center bg-[#0b0b10] text-[11px] uppercase tracking-[0.2em] text-white/35">
                            {date}
                          </div>
                        )}
                      </div>
                    </article>
                    );
                  })}
                </div>
              </div>

              <div className="h-1 w-full rounded-full bg-white/10">
                <div
                  ref={progressFillRef}
                  className="h-full rounded-full bg-gradient-to-r from-[#ff5138] to-[#ffd066]"
                  style={{ width: "6%" }}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
