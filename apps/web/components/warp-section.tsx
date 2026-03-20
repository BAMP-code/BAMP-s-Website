"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { BlackHoleHero } from "@/components/black-hole-hero";
import { StarfieldCanvas } from "@/components/starfield-canvas";
import { motion } from "@/lib/motion";

const OUTLOOK_COMPOSE_URL =
  "https://outlook.office.com/mail/deeplink/compose?to=pineda.bamp@gmail.com&subject=Portfolio%20Inquiry%20from%20Website";

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function WarpSection() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const heroContentRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef(0);
  const maxProgressRef = useRef(0);
  const introCompleteRef = useRef(false);
  const rafRef = useRef(0);
  const [showTopBar, setShowTopBar] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);

  // Scroll restoration — keep at page top on reload
  useEffect(() => {
    if (typeof window === "undefined") return;
    if ("scrollRestoration" in window.history) {
      window.history.scrollRestoration = "manual";
    }

    const navEntry = performance.getEntriesByType("navigation")[0] as
      | PerformanceNavigationTiming
      | undefined;
    if (navEntry?.type === "reload") {
      if (window.location.hash) {
        window.history.replaceState(null, "", window.location.pathname + window.location.search);
      }
      window.requestAnimationFrame(() => {
        window.scrollTo({ top: 0, left: 0, behavior: "auto" });
      });
    }
  }, []);

  // Reduced motion detection
  useEffect(() => {
    if (typeof window === "undefined") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduceMotion(query.matches);
    onChange();
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  const onIntroComplete = useCallback(() => {
    introCompleteRef.current = true;
    setShowTopBar(true);
  }, []);

  // Dampen scroll speed while the warp animation is active so the user
  // can't rush through it. We intercept wheel events, cancel the native
  // scroll, and re-apply at 40% speed.
  useEffect(() => {
    if (reduceMotion) return;

    const section = sectionRef.current;
    if (!section) return;

    const onWheel = (e: WheelEvent) => {
      // Only dampen while the section is on-screen and animation is playing
      const p = progressRef.current;
      if (p >= 1) return;
      if (!introCompleteRef.current) return;

      const rect = section.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) return;

      e.preventDefault();
      // Progressive damping: light at the start, heavier during the action.
      // 0–20% progress → 0.75x (barely noticeable)
      // 20–80% progress → ramps down to 0.35x (cinematic slow-down)
      // 80–100% progress → eases back to 0.65x (let the user finish)
      let factor: number;
      if (p < 0.2) {
        factor = 0.75;
      } else if (p < 0.8) {
        factor = 0.75 - (p - 0.2) * (0.4 / 0.6); // 0.75 → 0.35
      } else {
        factor = 0.35 + (p - 0.8) * (0.3 / 0.2); // 0.35 → 0.65
      }
      window.scrollBy(0, e.deltaY * factor);
    };

    section.addEventListener("wheel", onWheel, { passive: false });
    return () => section.removeEventListener("wheel", onWheel);
  }, [reduceMotion]);

  // Scroll-driven progress + hero content reveal
  useEffect(() => {
    if (reduceMotion) {
      progressRef.current = 1;
      return;
    }

    const section = sectionRef.current;
    const heroContent = heroContentRef.current;
    if (!section) return;

    let running = true;

    const update = () => {
      if (!running) return;

      if (!introCompleteRef.current) {
        progressRef.current = 0;
        rafRef.current = requestAnimationFrame(update);
        return;
      }

      const rect = section.getBoundingClientRect();
      const sectionHeight = section.offsetHeight;
      const viewportHeight = window.innerHeight;
      const scrolled = -rect.top;
      const maxScroll = sectionHeight - viewportHeight;
      const raw = maxScroll > 0 ? scrolled / maxScroll : 0;
      const nextProgress = Math.max(0, Math.min(1, raw));
      // One-way time jump: once advanced, never replay when user scrolls back up.
      maxProgressRef.current = Math.max(maxProgressRef.current, nextProgress);
      if (maxProgressRef.current > 0.98) {
        maxProgressRef.current = 1;
      }

      // Progress directly tracks scroll — no lerp.
      // The wheel damping above ensures the user can't rush the animation.
      progressRef.current = maxProgressRef.current;

      // Reveal hero content as the black hole fades away
      if (heroContent) {
        const contentT = smoothstep(0.40, 0.70, progressRef.current);
        heroContent.style.opacity = String(contentT);
        heroContent.style.transform = `translateY(${(1 - contentT) * 24}px)`;
      }

      // Stop the RAF loop once the animation is done and section is off-screen.
      if (progressRef.current >= 1 && rect.bottom < 0) {
        return;
      }

      rafRef.current = requestAnimationFrame(update);
    };

    rafRef.current = requestAnimationFrame(update);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [reduceMotion]);

  if (reduceMotion) {
    return (
      <div className="bg-black">
        <Header showTopBar={true} />
        <div className="flex min-h-[50vh] flex-col items-center justify-center px-6 text-center">
          <p className="text-xs uppercase tracking-[0.3em] text-accent/70">Bryan Pineda</p>
          <h2 className="mt-4 text-4xl font-bold text-white sm:text-6xl">BAMP</h2>
          <p className="mt-4 max-w-lg text-base leading-relaxed text-muted">
            Building intelligent embedded systems
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={sectionRef}
      className="relative bg-black"
      style={{ height: "320vh" }}
    >
      <Header showTopBar={showTopBar} />
      <div className="sticky top-0 h-screen overflow-hidden">
        {/* Hero content — revealed as the black hole moves down */}
        <div
          ref={heroContentRef}
          className="absolute inset-0 z-20 flex flex-col items-center justify-center pointer-events-none"
          style={{ opacity: 0 }}
        >
          <p className="text-xs uppercase tracking-[0.3em] text-accent/70 sm:text-sm">
            Bryan Pineda
          </p>
          <h2 className="mt-3 text-5xl font-bold tracking-tight text-white sm:text-7xl">
            BAMP
          </h2>
          <p className="mt-4 max-w-md text-center text-sm leading-relaxed text-muted sm:text-base">
            Building intelligent embedded systems
          </p>
          <div className="pointer-events-auto mt-8 flex gap-4">
            <a
              href="#projects"
              className="rounded-full border border-white/20 px-5 py-2 text-sm text-white/80 transition-colors hover:border-white hover:text-white"
            >
              View Projects
            </a>
            <a
              href="#about"
              className="rounded-full border border-accent/30 px-5 py-2 text-sm text-accent/80 transition-colors hover:border-accent hover:text-accent"
            >
              About Me
            </a>
          </div>
        </div>

        <BlackHoleHero progressRef={progressRef} onIntroComplete={onIntroComplete} />
        <StarfieldCanvas progressRef={progressRef} />
      </div>
    </div>
  );
}

function Header({ showTopBar }: { showTopBar: boolean }) {
  return (
    <header
      className={`fixed inset-x-0 top-0 z-40 border-b border-white/10 bg-black/90 transition-all duration-700 ${
        showTopBar ? "pointer-events-auto translate-y-0 opacity-100" : "pointer-events-none -translate-y-3 opacity-0"
      }`}
    >
      <div className="mx-auto flex w-[min(96vw,1800px)] items-center justify-between px-[clamp(10px,1.4vw,24px)] py-3 text-[13px]">
        <a href="/" className="font-semibold tracking-[0.02em] text-white no-underline">
          BAMP
        </a>
        <nav aria-label="Primary" className="flex items-center gap-5 text-muted sm:gap-7">
          <a href="#projects" className="hidden transition-colors hover:text-white sm:block">
            projects
          </a>
          <a href="#about" className="hidden transition-colors hover:text-white sm:block">
            about
          </a>
          <a
            href={OUTLOOK_COMPOSE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden text-accent-secondary transition-colors hover:text-accent-secondary/70 sm:block"
          >
            email me
          </a>
          <a href="#projects" className="transition-colors hover:text-white sm:hidden">
            projects
          </a>
          <a href="#about" className="transition-colors hover:text-white sm:hidden">
            about
          </a>
          <a
            href={OUTLOOK_COMPOSE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-secondary transition-colors hover:text-accent-secondary/70 sm:hidden"
          >
            email
          </a>
        </nav>
      </div>
      <div className="border-t border-white/10">
        <div className="mx-auto w-[min(96vw,1800px)] px-[clamp(10px,1.4vw,24px)] py-2 text-[11px] tracking-[0.08em] text-muted">
          <p>Bryan Pineda — Building intelligent embedded systems</p>
        </div>
      </div>
    </header>
  );
}
