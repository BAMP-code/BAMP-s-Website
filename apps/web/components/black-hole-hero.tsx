"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { CSSProperties } from "react";
import { motion } from "@/lib/motion";

const INTRO_ALWAYS_PLAY = true;

const { phases } = motion.warp;

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

type Props = {
  progressRef: React.RefObject<number>;
  onIntroComplete: () => void;
};

export function BlackHoleHero({ progressRef, onIntroComplete }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const turbulenceRef = useRef<SVGFETurbulenceElement>(null);
  const displacementRef = useRef<SVGFEDisplacementMapElement>(null);
  const blurRef = useRef<SVGFEGaussianBlurElement>(null);
  const glowGradientRef = useRef<SVGRadialGradientElement>(null);
  const charRefs = useRef<Map<string, HTMLSpanElement>>(new Map());
  const [shouldPlayIntro, setShouldPlayIntro] = useState<boolean | null>(null);

  const pseudoRandom = useCallback((index: number, salt: number) => {
    const value = Math.sin((index + 1) * 12.9898 + salt * 78.233) * 43758.5453;
    return value - Math.floor(value);
  }, []);

  // Determine intro state
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    if (INTRO_ALWAYS_PLAY) {
      setShouldPlayIntro(true);
      container.classList.remove("intro-static");
      container.classList.add("intro-playing");
      return;
    }

    setShouldPlayIntro(false);
    container.classList.add("intro-static");
    container.classList.remove("intro-playing");
  }, []);

  // Fire intro complete callback
  useEffect(() => {
    if (shouldPlayIntro === null) return;
    if (!shouldPlayIntro) {
      onIntroComplete();
      return;
    }

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      onIntroComplete();
      return;
    }

    const timer = window.setTimeout(() => {
      onIntroComplete();
    }, motion.warp.introDuration);

    return () => {
      window.clearTimeout(timer);
    };
  }, [shouldPlayIntro, onIntroComplete]);

  // Mouse-reactive SVG filter + scroll-reactive transforms
  useEffect(() => {
    const container = containerRef.current;
    const turbulence = turbulenceRef.current;
    const displacement = displacementRef.current;
    const blur = blurRef.current;
    const glowGradient = glowGradientRef.current;
    if (!container || !turbulence || !displacement || !blur || !glowGradient) return;

    let rafId = 0;
    let running = true;
    let targetInfluence = 0;
    let influence = 0;
    let targetX = 0.5;
    let targetY = 0.5;
    let x = 0.5;
    let y = 0.5;

    const onMove = (event: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const mx = (event.clientX - rect.left) / rect.width;
      const my = (event.clientY - rect.top) / rect.height;
      targetX = Math.max(0, Math.min(1, mx));
      targetY = Math.max(0, Math.min(1, my));

      const cx = rect.width / 2;
      const cy = rect.height / 2;
      const dx = event.clientX - rect.left - cx;
      const dy = event.clientY - rect.top - cy;
      const dist = Math.hypot(dx, dy);
      const maxDist = Math.min(rect.width, rect.height) * 0.48;
      targetInfluence = Math.max(0, 1 - dist / maxDist);
    };

    const onLeave = () => {
      targetInfluence = 0;
      targetX = 0.5;
      targetY = 0.5;
    };

    container.addEventListener("mousemove", onMove);
    container.addEventListener("mouseleave", onLeave);

    const tick = (time: number) => {
      if (!running) return;

      const p = progressRef.current ?? 0;
      if (p >= 1) {
        return;
      }

      // Black hole should feel like a distant time-jump object:
      // it shrinks and fades away instead of translating downward.
      const moveT = smoothstep(0.15, 0.70, p);
      const bhScale = 1 - moveT * 0.88;
      const bhOpacity = 1 - smoothstep(0.35, 0.82, p);

      if (p < 0.01) {
        container.style.opacity = "";
        container.style.transform = "";
      } else {
        container.style.opacity = String(bhOpacity);
        container.style.transform = `translateY(0px) scale(${Math.max(0.08, bhScale).toFixed(3)})`;
      }

      // Skip mouse interactivity once BH has moved significantly
      if (p > 0.40) {
        rafId = window.requestAnimationFrame(tick);
        return;
      }

      // Smooth mouse response
      influence += (targetInfluence - influence) * 0.08;
      x += (targetX - x) * 0.1;
      y += (targetY - y) * 0.1;

      const tx = time * 0.0008;
      const freqX = 0.011 + Math.sin(tx) * 0.0015 + influence * 0.006;
      const freqY = 0.018 + Math.cos(tx * 1.2) * 0.0018 + influence * 0.008;
      const scale = 10 + influence * 26;
      const blurValue = 2 + influence * 1.8;

      turbulence.setAttribute("baseFrequency", `${freqX.toFixed(4)} ${freqY.toFixed(4)}`);
      displacement.setAttribute("scale", scale.toFixed(2));
      blur.setAttribute("stdDeviation", blurValue.toFixed(2));

      const cx2 = 50 + (x - 0.5) * 22 * influence;
      const cy2 = 50 + (y - 0.5) * 22 * influence;
      glowGradient.setAttribute("cx", `${cx2.toFixed(2)}%`);
      glowGradient.setAttribute("cy", `${cy2.toFixed(2)}%`);

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);

    return () => {
      running = false;
      container.removeEventListener("mousemove", onMove);
      container.removeEventListener("mouseleave", onLeave);
      window.cancelAnimationFrame(rafId);
    };
    // progressRef is a stable ref — not a reactive value.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldPlayIntro]);

  // Letter gravity-suck animation
  useEffect(() => {
    const container = containerRef.current;
    if (
      !container ||
      shouldPlayIntro !== true ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return;
    }

    type Particle = {
      el: HTMLSpanElement;
      ox: number;
      oy: number;
      x: number;
      y: number;
      vx: number;
      vy: number;
      drag: number;
      swirl: number;
      startDist: number;
      startAt: number;
      angle: number;
      gone: boolean;
    };

    let rafId = 0;
    let startTimer = 0;
    let prevTime = 0;
    let particles: Particle[] = [];
    let horizonX = 0;
    let horizonY = 0;
    let absorbR = 0;

    const tick = (now: number) => {
      const dt = Math.min((now - prevTime) / 1000, 0.05);
      prevTime = now;
      let activeCount = 0;

      for (const p of particles) {
        if (p.gone) continue;

        if (now < p.startAt) {
          activeCount += 1;
          continue;
        }

        const dx = horizonX - p.x;
        const dy = horizonY - p.y;
        const dist = Math.max(0.001, Math.hypot(dx, dy));

        if (dist <= absorbR + 6) {
          p.gone = true;
          p.el.style.opacity = "0";
          p.el.style.visibility = "hidden";
          continue;
        }

        const nx = dx / dist;
        const ny = dy / dist;
        const tx = -ny;
        const ty = nx;
        // Strong radial pull that overwhelms swirl as letters get close
        const gravity = 78000 / (dist * dist + 5000);
        // Reduce swirl as letters approach to prevent orbiting
        const swirlFalloff = Math.min(1, dist / (absorbR * 4));
        const ax = nx * gravity + tx * gravity * p.swirl * swirlFalloff;
        const ay = ny * gravity + ty * gravity * p.swirl * swirlFalloff;

        p.vx = (p.vx + ax * dt * 1000) * p.drag;
        p.vy = (p.vy + ay * dt * 1000) * p.drag;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.angle += (p.vx - p.vy) * 0.011;

        const progress = Math.max(0, Math.min(1, (dist - absorbR) / (p.startDist - absorbR)));
        const scale = 0.2 + progress * 0.8;
        const blur = (1 - progress) * 1.5;
        const opacity = Math.max(0, Math.min(1, progress * 1.25));

        p.el.style.transform = `translate(${(p.x - p.ox).toFixed(2)}px, ${(p.y - p.oy).toFixed(2)}px) rotate(${p.angle.toFixed(2)}deg) scale(${scale.toFixed(3)})`;
        p.el.style.opacity = opacity.toFixed(3);
        p.el.style.filter = `blur(${blur.toFixed(2)}px)`;

        activeCount += 1;
      }

      if (activeCount > 0) {
        rafId = window.requestAnimationFrame(tick);
      } else {
        // All letters absorbed — hide the entire welcome text container
        const welcomeWrapper = container.querySelector(".pointer-events-none");
        if (welcomeWrapper instanceof HTMLElement) {
          welcomeWrapper.style.visibility = "hidden";
        }
      }
    };

    const startSuck = () => {
      const svg = container.querySelector("svg");
      if (!(svg instanceof SVGElement)) return;

      const svgRect = svg.getBoundingClientRect();
      horizonX = svgRect.left + svgRect.width * 0.5;
      horizonY = svgRect.top + svgRect.height * 0.5;
      const horizonR = (svgRect.width * 62) / 420;
      absorbR = Math.max(30, Math.min(horizonR * 0.55, 110));

      const chars = Array.from(charRefs.current.values()).sort((a, b) => {
        const ao = Number(a.dataset.charOrder ?? "0");
        const bo = Number(b.dataset.charOrder ?? "0");
        return ao - bo;
      });

      const startNow = performance.now();
      particles = chars.map((el) => {
        const rect = el.getBoundingClientRect();
        const ox = rect.left + rect.width * 0.5;
        const oy = rect.top + rect.height * 0.5;
        const dx = horizonX - ox;
        const dy = horizonY - oy;
        const dist = Math.hypot(dx, dy);
        const seed = Number(el.dataset.seed ?? "0.5");
        const seed2 = (seed * 1.37) % 1;
        const seed3 = (seed * 1.91) % 1;
        const seed4 = (seed * 2.53) % 1;

        el.style.transform = "translate(0px, 0px) rotate(0deg) scale(1)";
        el.style.opacity = "1";
        el.style.filter = "blur(0px)";

        return {
          el,
          ox,
          oy,
          x: ox,
          y: oy,
          vx: (seed2 - 0.5) * 18,
          vy: (seed3 - 0.5) * 18,
          drag: 0.965 + seed4 * 0.015,
          swirl: (seed - 0.5) * 2.2,
          startDist: Math.max(dist, absorbR + 10),
          startAt: startNow + 240 + seed3 * 640,
          angle: (seed - 0.5) * 26,
          gone: false,
        };
      });

      prevTime = startNow;
      rafId = window.requestAnimationFrame(tick);
    };

    startTimer = window.setTimeout(startSuck, 4700);

    return () => {
      window.clearTimeout(startTimer);
      window.cancelAnimationFrame(rafId);
    };
  }, [shouldPlayIntro]);

  const renderWelcomeText = (text: string, salt: number, orderOffset: number) => {
    const chars = Array.from(text);

    return chars.map((char, index) => {
      const seed = pseudoRandom(index, salt + 0.37);
      const charId = `${salt}-${index}`;

      return (
        <span
          key={`${charId}-${char}`}
          className="welcome-char"
          data-char-order={orderOffset + index}
          data-seed={seed.toFixed(5)}
          ref={(node) => {
            if (node) {
              charRefs.current.set(charId, node);
            } else {
              charRefs.current.delete(charId);
            }
          }}
        >
          {char === " " ? "\u00A0" : char}
        </span>
      );
    });
  };

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-10 flex items-center justify-center"
    >
      {shouldPlayIntro !== false && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 z-30 -translate-x-1/2 -translate-y-1/2">
          <h1
            className={`text-center text-[clamp(1.4rem,4vw,3.1rem)] font-semibold tracking-[0.03em] text-[#f5f7ff] ${
              shouldPlayIntro === null ? "opacity-0" : "opacity-100"
            }`}
          >
            <span
              className="welcome-line welcome-line-top"
              style={{ "--line-delay": shouldPlayIntro ? "1s" : "0ms" } as CSSProperties}
            >
              <span className="welcome-text">
                {renderWelcomeText("Welcome To", 3, 0)}
              </span>
              <span className="welcome-box" aria-hidden="true" />
            </span>
            <span
              className="welcome-line welcome-line-bottom"
              style={{ "--line-delay": shouldPlayIntro ? "1.58s" : "0ms" } as CSSProperties}
            >
              <span className="welcome-text">
                {renderWelcomeText("My Website", 17, 100)}
              </span>
              <span className="welcome-box" aria-hidden="true" />
            </span>
          </h1>
        </div>
      )}

      <div className="relative z-10 w-[min(88vw,980px)]">
        <svg
          viewBox="0 0 420 420"
          className="h-auto w-full"
          role="img"
          aria-label="Interactive black hole"
        >
          <defs>
            <radialGradient id="bgVignette" cx="50%" cy="50%" r="60%">
              <stop offset="0%" stopColor="#1a0600" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0" />
            </radialGradient>

            <radialGradient id="gasGlow" ref={glowGradientRef} cx="50%" cy="50%" r="56%">
              <stop offset="0%" stopColor="#ffb15d" stopOpacity="0.92" />
              <stop offset="46%" stopColor="#ff5a00" stopOpacity="0.54" />
              <stop offset="100%" stopColor="#ff2f00" stopOpacity="0" />
            </radialGradient>

            <linearGradient id="gasBand" x1="18%" y1="26%" x2="88%" y2="78%">
              <stop offset="0%" stopColor="#ffd08b" />
              <stop offset="55%" stopColor="#ff6b00" />
              <stop offset="100%" stopColor="#7a1f00" />
            </linearGradient>

            <filter id="gasDistort" x="-50%" y="-50%" width="200%" height="200%">
              <feTurbulence
                ref={turbulenceRef}
                type="fractalNoise"
                baseFrequency="0.011 0.018"
                numOctaves="2"
                seed="8"
                result="noise"
              />
              <feDisplacementMap
                ref={displacementRef}
                in="SourceGraphic"
                in2="noise"
                scale="10"
                xChannelSelector="R"
                yChannelSelector="G"
                result="distorted"
              />
              <feGaussianBlur
                ref={blurRef}
                in="distorted"
                stdDeviation="2"
                result="soft"
              />
              <feMerge>
                <feMergeNode in="soft" />
                <feMergeNode in="distorted" />
              </feMerge>
            </filter>
          </defs>

          <circle cx="210" cy="210" r="180" fill="url(#bgVignette)" />
          <circle cx="210" cy="210" r="122" fill="url(#gasGlow)" opacity="0.82" />

          <ellipse
            cx="210"
            cy="210"
            rx="126"
            ry="90"
            fill="none"
            stroke="url(#gasBand)"
            strokeWidth="30"
            filter="url(#gasDistort)"
            opacity="0.88"
            transform="rotate(-12 210 210)"
          />

          <ellipse
            cx="210"
            cy="210"
            rx="112"
            ry="80"
            fill="none"
            stroke="url(#gasBand)"
            strokeWidth="18"
            filter="url(#gasDistort)"
            opacity="0.74"
            transform="rotate(-12 210 210)"
          />

          <circle cx="210" cy="210" r="62" fill="#020202" />
        </svg>
      </div>
    </div>
  );
}
