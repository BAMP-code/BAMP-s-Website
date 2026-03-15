"use client";

import { useEffect, useRef } from "react";

export function BlackHoleHero() {
  const sectionRef = useRef<HTMLElement>(null);
  const turbulenceRef = useRef<SVGFETurbulenceElement>(null);
  const displacementRef = useRef<SVGFEDisplacementMapElement>(null);
  const blurRef = useRef<SVGFEGaussianBlurElement>(null);
  const glowGradientRef = useRef<SVGRadialGradientElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const turbulence = turbulenceRef.current;
    const displacement = displacementRef.current;
    const blur = blurRef.current;
    const glowGradient = glowGradientRef.current;
    if (!section || !turbulence || !displacement || !blur || !glowGradient) {
      return;
    }

    let rafId = 0;
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
      const maxDist = Math.min(rect.width, rect.height) * 0.48;
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

      // Smooth response for premium motion.
      influence += (targetInfluence - influence) * 0.08;
      x += (targetX - x) * 0.1;
      y += (targetY - y) * 0.1;

      const tx = time * 0.0008;
      const freqX = 0.011 + Math.sin(tx) * 0.0015 + influence * 0.006;
      const freqY = 0.018 + Math.cos(tx * 1.2) * 0.0018 + influence * 0.008;
      const scale = 10 + influence * 26;
      const blurValue = 2 + influence * 1.8;

      turbulence.setAttribute(
        "baseFrequency",
        `${freqX.toFixed(4)} ${freqY.toFixed(4)}`,
      );
      displacement.setAttribute("scale", scale.toFixed(2));
      blur.setAttribute("stdDeviation", blurValue.toFixed(2));

      // Pull glow center gently toward cursor when nearby.
      const cx = 50 + (x - 0.5) * 22 * influence;
      const cy = 50 + (y - 0.5) * 22 * influence;
      glowGradient.setAttribute("cx", `${cx.toFixed(2)}%`);
      glowGradient.setAttribute("cy", `${cy.toFixed(2)}%`);

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);

    return () => {
      running = false;
      section.removeEventListener("mousemove", onMove);
      section.removeEventListener("mouseleave", onLeave);
      window.cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative flex h-screen items-center justify-center overflow-hidden bg-black"
    >
      <div className="relative z-10 w-[min(82vw,620px)]">
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
                numOctaves="3"
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
    </section>
  );
}
