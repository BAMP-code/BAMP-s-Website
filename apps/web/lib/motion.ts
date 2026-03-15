export const motion = {
  slide: {
    duration: 520,
    offset: 44,
    ease: [0.22, 1, 0.36, 1] as const,
  },
  sectionReveal: {
    duration: 820,
    translateY: 26,
    threshold: 0.35,
  },
  marquee: {
    duration: 60_000,
    distance: 3800,
  },
  aboutReveal: {
    duration: 760,
    translateY: 24,
    threshold: 0.32,
  },
  fadeIn: {
    duration: 260,
    translateY: 5,
  },
} as const;
