"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { BlackHoleHero } from "@/components/black-hole-hero";
import { StarfieldCanvas } from "@/components/starfield-canvas";
import { ProjectsGrid } from "@/components/projects-grid";
import { motion } from "@/lib/motion";

const OUTLOOK_COMPOSE_URL =
  "https://outlook.office.com/mail/deeplink/compose?to=pineda.bamp@gmail.com&subject=Portfolio%20Inquiry%20from%20Website";

export function WarpSection() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef(0);
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

  // Scroll-driven progress
  useEffect(() => {
    if (reduceMotion) {
      progressRef.current = 1;
      return;
    }

    const section = sectionRef.current;
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
      progressRef.current = Math.max(0, Math.min(1, raw));

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
      <div id="projects" className="scroll-mt-32 bg-black">
        <Header showTopBar={true} />
        <ProjectsGrid progressRef={progressRef} />
      </div>
    );
  }

  return (
    <div
      ref={sectionRef}
      id="projects"
      className="relative scroll-mt-32 bg-black"
      style={{ height: `${motion.warp.runwayVh}vh` }}
    >
      <Header showTopBar={showTopBar} />
      <div className="sticky top-0 h-screen overflow-hidden">
        <BlackHoleHero progressRef={progressRef} onIntroComplete={onIntroComplete} />
        <StarfieldCanvas progressRef={progressRef} />
        <ProjectsGrid progressRef={progressRef} />
      </div>
    </div>
  );
}

function Header({ showTopBar }: { showTopBar: boolean }) {
  return (
    <header
      className={`pointer-events-auto fixed inset-x-0 top-0 z-40 border-b border-white/10 bg-black/70 backdrop-blur-md transition-all duration-700 ${
        showTopBar ? "translate-y-0 opacity-100" : "-translate-y-3 opacity-0"
      }`}
    >
      <div className="mx-auto flex w-[min(96vw,1800px)] items-center justify-between px-[clamp(10px,1.4vw,24px)] py-3 text-[13px]">
        <a href="/" className="font-semibold tracking-[0.02em] text-white no-underline">
          BAMP
        </a>
        <nav aria-label="Primary" className="hidden items-center gap-7 text-white/75 sm:flex">
          <a href="#projects" className="transition-colors hover:text-white">
            projects
          </a>
          <a href="#about" className="transition-colors hover:text-white">
            about
          </a>
          <a
            href={OUTLOOK_COMPOSE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#ff4f4f] transition-colors hover:text-[#ff8b8b]"
          >
            email me
          </a>
        </nav>
      </div>
      <div className="border-t border-white/10">
        <div className="mx-auto w-[min(96vw,1800px)] px-[clamp(10px,1.4vw,24px)] py-2 text-[11px] tracking-[0.08em] text-white/45">
          <p>Bryan Pineda — Embedded systems and intelligent products</p>
        </div>
      </div>
    </header>
  );
}
