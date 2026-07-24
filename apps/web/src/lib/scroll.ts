import Lenis from "lenis";

// Smooth scrolling for the editorial site. Lenis owns the wheel/touch
// smoothing and in-page anchor navigation; disabled under reduced motion.

let lenis: Lenis | null = null;
let initialized = false;

export function initScroll() {
  if (typeof window === "undefined" || initialized) return;
  initialized = true;

  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  // Reduced-motion users keep native scrolling; anchors jump instantly.
  if (reduceMotion) return;

  lenis = new Lenis({
    duration: 1.1,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    smoothWheel: true,
  });

  const raf = (time: number) => {
    lenis?.raf(time);
    requestAnimationFrame(raf);
  };
  requestAnimationFrame(raf);

  // Route in-page anchor clicks through Lenis so smoothing is consistent.
  // Sections carry `scroll-mt-*` (Lenis honors scroll-margin), which
  // clears the sticky nav — so no extra offset here.
  document
    .querySelectorAll<HTMLAnchorElement>('a[href^="#"]')
    .forEach((anchor) => {
      anchor.addEventListener("click", (event) => {
        // Let modified / non-primary clicks (open in new tab, etc.) behave natively.
        if (
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey
        )
          return;
        const hash = anchor.getAttribute("href");
        if (!hash || hash === "#") return;
        const target = document.querySelector(hash);
        if (!(target instanceof HTMLElement)) return;
        event.preventDefault();
        // Keep the fragment in the URL/history so Back works and links are shareable.
        history.pushState(null, "", hash);
        lenis?.scrollTo(target, {
          onComplete: () => {
            // Move keyboard focus to the target so skip links (e.g. #work) work.
            target.setAttribute("tabindex", "-1");
            target.focus({ preventScroll: true });
          },
        });
      });
    });
}
