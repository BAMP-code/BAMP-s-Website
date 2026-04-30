import Lenis from "lenis";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

// Scroll system. One source of truth for "where in the page are we"
// (scrollProgressRef.current ∈ [0, 1]). The R3F scene reads it each
// frame; ScrollTrigger reads via Lenis's tick.

gsap.registerPlugin(ScrollTrigger);

export const scrollProgressRef = { current: 0 };

let lenis: Lenis | null = null;
let initialized = false;

export function initScroll() {
  if (typeof window === "undefined" || initialized) return;
  initialized = true;

  // Reduced-motion users get native scroll, no smoothing.
  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  if (reduceMotion) {
    const onScroll = () => {
      const max =
        document.documentElement.scrollHeight - window.innerHeight;
      scrollProgressRef.current = max > 0 ? window.scrollY / max : 0;
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return;
  }

  lenis = new Lenis({
    duration: 1.1,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    smoothWheel: true,
  });

  lenis.on("scroll", ({ scroll, limit }: { scroll: number; limit: number }) => {
    scrollProgressRef.current = limit > 0 ? scroll / limit : 0;
    ScrollTrigger.update();
  });

  // Drive Lenis from GSAP's ticker so they share one rAF.
  gsap.ticker.add((time) => {
    lenis?.raf(time * 1000);
  });
  gsap.ticker.lagSmoothing(0);
}
