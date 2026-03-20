"use client";

import Image from "next/image";
import { useEffect, useRef } from "react";
import { projects } from "@/content/projects";
import { motion } from "@/lib/motion";

const { phases, cardStagger } = motion.warp;

const FEATURED_INDICES = new Set([0, 4, 7]);

const CATEGORY_LABELS: Record<string, string> = {
  cs: "CS",
  "ee-me": "EE / ME",
  drawings: "Drawings",
};

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function VideoPlaceholder({ src, alt }: { src: string; alt: string }) {
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

type Props = {
  progressRef: React.RefObject<number>;
};

export function ProjectsGrid({ progressRef }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);
  const rafRef = useRef(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let running = true;

    const animate = () => {
      if (!running) return;

      const p = progressRef.current ?? 0;

      // Container visibility
      const containerOpacity = smoothstep(phases.reveal[0], phases.reveal[1], p);
      container.style.opacity = String(containerOpacity);

      if (containerOpacity < 0.01) {
        container.style.visibility = "hidden";
        rafRef.current = requestAnimationFrame(animate);
        return;
      }
      container.style.visibility = "visible";

      // Individual card stagger
      let allSettled = true;
      for (let i = 0; i < cardRefs.current.length; i++) {
        const card = cardRefs.current[i];
        if (!card) continue;

        const cardStart = 0.74 + i * cardStagger;
        const cardEnd = cardStart + 0.18;
        const t = smoothstep(cardStart, cardEnd, p);

        if (t < 1) allSettled = false;

        const scale = 0.85 + t * 0.15;
        const opacity = t;
        card.style.transform = `scale(${scale})`;
        card.style.opacity = String(opacity);

        if (t >= 1) {
          card.style.willChange = "auto";
        } else if (p > phases.reveal[0] - 0.05) {
          card.style.willChange = "transform, opacity";
        }
      }

      // Keep running if cards are still animating or user might scroll back
      if (allSettled && p > 0.95) {
        // All cards fully revealed and past scroll range — stop RAF
        return;
      }

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [progressRef]);

  return (
    <div
      ref={containerRef}
      className="warp-projects absolute inset-0 z-30 flex items-center justify-center overflow-y-auto"
      style={{ opacity: 0, visibility: "hidden" }}
    >
      <div className="mx-auto w-full max-w-[1400px] px-4 py-16">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((project, index) => {
            const isFeatured = FEATURED_INDICES.has(index);
            return (
              <div
                key={project.id}
                ref={(el) => { cardRefs.current[index] = el; }}
                className={`overflow-hidden rounded-xl ${
                  isFeatured ? "sm:col-span-2 sm:row-span-2" : ""
                }`}
                style={{ opacity: 0, transform: "scale(0.85)" }}
              >
                <div className="group relative h-full min-h-[200px]">
                  <div
                    className={`relative w-full overflow-hidden ${
                      isFeatured ? "h-[320px] sm:h-[420px]" : "h-[200px] sm:h-[260px]"
                    }`}
                  >
                    {project.media.type === "video" ? (
                      <VideoPlaceholder
                        src={project.media.src}
                        alt={project.media.alt}
                      />
                    ) : (
                      <Image
                        src={project.media.src}
                        alt={project.media.alt}
                        width={project.media.width}
                        height={project.media.height}
                        className="h-full w-full object-cover"
                        loading="lazy"
                        sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                        quality={75}
                      />
                    )}
                  </div>

                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-4 pt-12">
                    <span className="mb-1.5 inline-block rounded-full bg-white/10 px-2.5 py-0.5 text-[10px] uppercase tracking-[0.14em] text-white/70">
                      {CATEGORY_LABELS[project.category] ?? project.category}
                    </span>
                    <h3 className="text-lg font-semibold text-white">
                      {project.title}
                    </h3>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
