"use client";

import { useEffect, useRef, useCallback } from "react";
import { motion } from "@/lib/motion";

const { focalLength, phases } = motion.warp;

type Star = {
  x: number;
  y: number;
  baseZ: number;
  z: number;
  hue: number;
  brightness: number;
  prevSX: number;
  prevSY: number;
};

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function createStars(count: number): Star[] {
  const stars: Star[] = [];
  for (let i = 0; i < count; i++) {
    // Moderate baseZ spreads stars across the screen initially.
    // As warpSpeed increases and z drops, they streak outward.
    const baseZ = 400 + Math.random() * 1200;
    stars.push({
      x: (Math.random() - 0.5) * 3000,
      y: (Math.random() - 0.5) * 3000,
      baseZ,
      z: baseZ,
      hue: 210 + Math.random() * 40,
      brightness: Math.random(),
      prevSX: 0,
      prevSY: 0,
    });
  }
  return stars;
}

type Props = {
  progressRef: React.RefObject<number>;
};

export function StarfieldCanvas({ progressRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const starsRef = useRef<Star[] | null>(null);
  const rafRef = useRef(0);

  const getStarCount = useCallback(() => {
    if (typeof window === "undefined") return motion.warp.starCount;
    return window.innerWidth < 768
      ? motion.warp.starCountMobile
      : motion.warp.starCount;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio, 2);

    const resize = () => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();

    if (!starsRef.current) {
      starsRef.current = createStars(getStarCount());
    }
    const stars = starsRef.current;

    // Initialize prev screen positions to center
    const initCX = canvas.clientWidth / 2;
    const initCY = canvas.clientHeight / 2;
    for (const s of stars) {
      s.prevSX = initCX;
      s.prevSY = initCY;
    }

    let running = true;
    let wasInactive = true;
    let finished = false;

    const draw = () => {
      if (!running) return;

      const p = progressRef.current ?? 0;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;

      // Stop drawing after stars have faded out
      if (p > 0.92) {
        if (!wasInactive) {
          ctx.clearRect(0, 0, w, h);
          wasInactive = true;
        }
        if (!finished) {
          finished = true;
          return;
        }
        return;
      }
      finished = false;

      // Reset prev positions to current projected positions when
      // entering active range, to prevent streaks from stale coords
      if (wasInactive) {
        wasInactive = false;
        const cx0 = w / 2;
        const cy0 = h / 2;
        for (const s of stars) {
          const z = Math.max(s.z, 1);
          s.prevSX = cx0 + s.x * (focalLength / z);
          s.prevSY = cy0 + s.y * (focalLength / z);
        }
      }

      const cx = w / 2;
      const cy = h / 2;

      ctx.clearRect(0, 0, w, h);

      // Phase calculations — stars visible from the start
      const warpSpeed = smoothstep(phases.warpAccel[0], phases.warpAccel[1], p);
      // Fade to a dim level but keep some stars visible
      const fadeOut = 1 - smoothstep(phases.reveal[0], phases.reveal[1], p) * 0.85;
      // Gradual blend from dots to streaks
      const streakBlend = smoothstep(0.25, 0.45, p);

      for (const star of stars) {
        star.z = star.baseZ * (1 - warpSpeed * 0.95);
        const z = Math.max(star.z, 1);

        const sx = cx + star.x * (focalLength / z);
        const sy = cy + star.y * (focalLength / z);

        // Closeness factor (brighter when closer)
        const closeness = Math.max(0, Math.min(1, 1 - z / 1600));
        // Faint at rest (warpSpeed~0), brighten as warp kicks in
        const intensity = lerp(0.15, 1, warpSpeed);
        const alpha = fadeOut * intensity * (0.4 + closeness * 0.6);

        if (alpha < 0.01) {
          star.prevSX = sx;
          star.prevSY = sy;
          continue;
        }

        // Draw dots (fading out) and streaks (fading in) during blend
        if (streakBlend < 1) {
          const dotAlpha = alpha * (1 - streakBlend);
          const radius = lerp(1, 2, closeness);
          ctx.beginPath();
          ctx.arc(sx, sy, radius, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${star.hue}, 60%, ${70 + star.brightness * 30}%, ${dotAlpha})`;
          ctx.fill();
        }

        if (streakBlend > 0 && warpSpeed > 0.005) {
          const streakAlpha = alpha * streakBlend;
          const lw = lerp(1, 2.5, closeness);
          ctx.beginPath();
          ctx.moveTo(star.prevSX, star.prevSY);
          ctx.lineTo(sx, sy);
          ctx.strokeStyle = `hsla(${star.hue}, 60%, ${70 + star.brightness * 30}%, ${streakAlpha})`;
          ctx.lineWidth = lw;
          ctx.stroke();
        }

        star.prevSX = sx;
        star.prevSY = sy;
      }

      // White radial flash at peak warp
      if (p >= phases.flash[0] && p <= phases.flash[1] + 0.05) {
        const flashT = smoothstep(phases.flash[0], 0.60, p);
        const flashFade = 1 - smoothstep(0.60, phases.flash[1] + 0.05, p);
        const flashAlpha = flashT * flashFade * 0.25;

        if (flashAlpha > 0.005) {
          const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.min(w, h) * 0.5);
          grad.addColorStop(0, `rgba(255, 255, 255, ${flashAlpha})`);
          grad.addColorStop(1, "rgba(255, 255, 255, 0)");
          ctx.fillStyle = grad;
          ctx.fillRect(0, 0, w, h);
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    window.addEventListener("resize", resize);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [progressRef, getStarCount]);

  return (
    <canvas
      ref={canvasRef}
      className="warp-canvas absolute inset-0 z-[5] h-full w-full pointer-events-none"
      aria-hidden="true"
    />
  );
}
