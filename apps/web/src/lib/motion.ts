import { animate, createTimeline, onScroll, stagger } from "animejs";

// anime.js-driven motion: a "system boot" intro for the hero, scroll-
// triggered reveals for section content, and count-up stat numbers.
// Everything here only runs once .js-motion is on <html> (see the inline
// script in Base.astro), which itself is gated on prefers-reduced-motion —
// so reduced-motion users get the plain, fully-visible CSS state and never
// pay for this module doing anything.

let initialized = false;

export function initMotion() {
  if (typeof window === "undefined" || initialized) return;
  initialized = true;

  if (!document.documentElement.classList.contains("js-motion")) return;

  bootIntro();
  scrollReveals();
  statCounters();
}

function bootIntro() {
  const corners = document.querySelectorAll<HTMLElement>(".hud-corner");
  const status = document.querySelector<HTMLElement>(".hero-status");
  const headline = document.querySelector<HTMLElement>(".hero-headline");
  const lead = document.querySelector<HTMLElement>(".hero-lead");
  const metaItems = document.querySelectorAll<HTMLElement>(".hero-meta > div");

  const tl = createTimeline({ defaults: { ease: "outExpo" } });

  if (corners.length) {
    tl.add(
      corners,
      { opacity: [0, 0.85], scale: [0.6, 1], duration: 500, delay: stagger(60) },
      0,
    );
  }
  if (status) {
    tl.add(status, { opacity: [0, 1], translateY: [12, 0], duration: 500 }, 120);
  }
  if (headline) {
    tl.add(headline, { opacity: [0, 1], translateY: [28, 0], duration: 700 }, 220);
  }
  if (lead) {
    tl.add(lead, { opacity: [0, 1], translateY: [18, 0], duration: 600 }, 420);
  }
  if (metaItems.length) {
    tl.add(
      metaItems,
      { opacity: [0, 1], translateY: [14, 0], duration: 500, delay: stagger(80) },
      560,
    );
  }
}

function scrollReveals() {
  document.querySelectorAll<HTMLElement>(".reveal").forEach((el) => {
    animate(el, {
      opacity: [0, 1],
      translateY: [24, 0],
      duration: 700,
      ease: "outExpo",
      autoplay: onScroll({ target: el, enter: "bottom-=10% top", repeat: false }),
    });
  });
}

function statCounters() {
  document.querySelectorAll<HTMLElement>("[data-count-to]").forEach((el) => {
    const to = Number(el.dataset.countTo);
    if (!Number.isFinite(to)) return;
    const suffix = el.dataset.suffix ?? "";
    const pad = Number(el.dataset.pad ?? 0);
    const state = { value: 0 };
    animate(state, {
      value: to,
      duration: 1100,
      ease: "outExpo",
      autoplay: onScroll({ target: el, enter: "bottom-=10% top", repeat: false }),
      onUpdate: () => {
        const n = Math.round(state.value);
        el.textContent = `${pad ? String(n).padStart(pad, "0") : n}${suffix}`;
      },
    });
  });
}
