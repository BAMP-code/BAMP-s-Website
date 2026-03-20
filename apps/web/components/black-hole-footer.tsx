"use client";

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

type BlackHoleFooterProps = {
  overlay?: ReactNode;
};

export function BlackHoleFooter({ overlay }: BlackHoleFooterProps) {
  const sectionRef = useRef<HTMLElement>(null);
  const turbulenceRef = useRef<SVGFETurbulenceElement>(null);
  const displacementRef = useRef<SVGFEDisplacementMapElement>(null);
  const blurRef = useRef<SVGFEGaussianBlurElement>(null);
  const glowGradientRef = useRef<SVGRadialGradientElement>(null);
  const rafRef = useRef(0);
  const inViewportRef = useRef(false);

  // Track viewport visibility for RAF gating
  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        inViewportRef.current = entry.isIntersecting;
      },
      { threshold: 0.05, rootMargin: "0px 0px 24% 0px" },
    );

    observer.observe(section);
    return () => observer.disconnect();
  }, []);

  // Gentle idle animation for the turbulence filter
  useEffect(() => {
    const section = sectionRef.current;
    const turbulence = turbulenceRef.current;
    const displacement = displacementRef.current;
    const blur = blurRef.current;
    const glowGradient = glowGradientRef.current;
    if (!section || !turbulence || !displacement || !blur || !glowGradient) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let running = true;
    let targetInfluence = 0;
    let influence = 0;
    let targetX = 0.5;
    let targetY = 0.5;
    let x = 0.5;
    let y = 0.5;

    const onMove = (event: MouseEvent) => {
      const rect = section.getBoundingClientRect();
      const mx = (event.clientX - rect.left) / rect.width;
      const my = (event.clientY - rect.top) / rect.height;
      targetX = Math.max(0, Math.min(1, mx));
      targetY = Math.max(0, Math.min(1, my));

      const cx = rect.width / 2;
      const cy = rect.height / 2;
      const dx = event.clientX - rect.left - cx;
      const dy = event.clientY - rect.top - cy;
      const dist = Math.hypot(dx, dy);
      const maxDist = Math.min(rect.width, rect.height) * 0.56;
      targetInfluence = Math.max(0, 1 - dist / maxDist);
    };

    const onLeave = () => {
      targetInfluence = 0;
      targetX = 0.5;
      targetY = 0.5;
    };

    section.addEventListener("mousemove", onMove);
    section.addEventListener("mouseleave", onLeave);

    const tick = (time: number) => {
      if (!running) return;
      rafRef.current = requestAnimationFrame(tick);

      // Skip expensive SVG filter updates when not in viewport
      if (!inViewportRef.current) return;

      influence += (targetInfluence - influence) * 0.08;
      x += (targetX - x) * 0.1;
      y += (targetY - y) * 0.1;

      const tx = time * 0.00045;
      const freqX = 0.011 + Math.sin(tx) * 0.0012 + influence * 0.0045;
      const freqY = 0.018 + Math.cos(tx * 1.2) * 0.0014 + influence * 0.006;
      const distScale = 10 + influence * 18;
      const blurValue = 2 + influence * 1.4;

      turbulence.setAttribute("baseFrequency", `${freqX.toFixed(4)} ${freqY.toFixed(4)}`);
      displacement.setAttribute("scale", distScale.toFixed(2));
      blur.setAttribute("stdDeviation", blurValue.toFixed(2));

      const cx2 = 50 + (x - 0.5) * 18 * influence;
      const cy2 = 50 + (y - 0.5) * 18 * influence;
      glowGradient.setAttribute("cx", `${cx2.toFixed(2)}%`);
      glowGradient.setAttribute("cy", `${cy2.toFixed(2)}%`);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
      section.removeEventListener("mousemove", onMove);
      section.removeEventListener("mouseleave", onLeave);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section
      ref={sectionRef}
      aria-hidden={overlay ? undefined : true}
      className="relative overflow-hidden bg-black"
      style={{ height: "clamp(320px, 56vw, 720px)" }}
    >
      {overlay ? (
        <div className="pointer-events-none absolute inset-x-0 top-[clamp(20px,4vw,72px)] z-30">
          <div className="pointer-events-auto">{overlay}</div>
        </div>
      ) : null}

      <div className="absolute inset-x-0 bottom-0 flex items-center justify-center"
        style={{ transform: "translateY(58%)" }}
      >
        <div className="w-full">
          <svg
            viewBox="0 0 420 420"
            className="h-auto w-full"
            style={{ transform: "scaleX(1.28)" }}
          >
            <defs>
              <radialGradient id="footerBgVignette" cx="50%" cy="50%" r="60%">
                <stop offset="0%" stopColor="#1a0600" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#000000" stopOpacity="0" />
              </radialGradient>

              <radialGradient id="footerGasGlow" ref={glowGradientRef} cx="50%" cy="50%" r="56%">
                <stop offset="0%" stopColor="#ffb15d" stopOpacity="0.92" />
                <stop offset="46%" stopColor="#ff5a00" stopOpacity="0.54" />
                <stop offset="100%" stopColor="#ff2f00" stopOpacity="0" />
              </radialGradient>

              <linearGradient id="footerGasBand" x1="18%" y1="26%" x2="88%" y2="78%">
                <stop offset="0%" stopColor="#ffd08b" />
                <stop offset="55%" stopColor="#ff6b00" />
                <stop offset="100%" stopColor="#7a1f00" />
              </linearGradient>

              <filter id="footerGasDistort" x="-50%" y="-50%" width="200%" height="200%">
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

            <circle cx="210" cy="210" r="180" fill="url(#footerBgVignette)" />
            <circle cx="210" cy="210" r="122" fill="url(#footerGasGlow)" opacity="0.82" />

            <ellipse
              cx="210" cy="210" rx="126" ry="90"
              fill="none" stroke="url(#footerGasBand)" strokeWidth="30"
              filter="url(#footerGasDistort)" opacity="0.88"
              transform="rotate(-12 210 210)"
            />
            <ellipse
              cx="210" cy="210" rx="112" ry="80"
              fill="none" stroke="url(#footerGasBand)" strokeWidth="18"
              filter="url(#footerGasDistort)" opacity="0.74"
              transform="rotate(-12 210 210)"
            />

            <circle cx="210" cy="210" r="62" fill="#020202" />
          </svg>
        </div>
      </div>
    </section>
  );
}
