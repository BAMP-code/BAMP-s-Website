import Lenis from "lenis";

// Smooth-scroll for the v2 editorial site (Lenis only — the old
// gsap/ScrollTrigger + camera coupling belonged to the 3D descent and
// is gone from the homepage). `scrollProgressRef` is kept exported for
// the retained-but-unmounted scene components.

export const scrollProgressRef = { current: 0 };

let lenis: Lenis | null = null;
let initialized = false;

function setProgress(scroll: number, limit: number) {
  scrollProgressRef.current = limit > 0 ? scroll / limit : 0;
}

export function initScroll() {
  if (typeof window === "undefined" || initialized) return;
  initialized = true;

  // Reduced-motion users get native scroll, no smoothing.
  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  if (reduceMotion) {
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(window.scrollY, max);
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
    setProgress(scroll, limit);
  });

  const raf = (time: number) => {
    lenis?.raf(time);
    requestAnimationFrame(raf);
  };
  requestAnimationFrame(raf);

  // Route in-page anchor clicks through Lenis so smoothing is consistent
  // (offset clears the sticky nav). Without JS, native jumps still work.
  document
    .querySelectorAll<HTMLAnchorElement>('a[href^="#"]')
    .forEach((anchor) => {
      anchor.addEventListener("click", (event) => {
        const hash = anchor.getAttribute("href");
        if (!hash || hash === "#") return;
        const target = document.querySelector(hash);
        if (!target) return;
        event.preventDefault();
        // Sections already carry `scroll-mt-20` (Lenis honors scroll-margin),
        // which clears the sticky nav — so no extra offset here.
        lenis?.scrollTo(target as HTMLElement);
      });
    });
}
