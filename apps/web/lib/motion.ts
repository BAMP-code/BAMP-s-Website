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
  warp: {
    /** Total scroll runway multiplier (5 × viewport height) */
    runwayVh: 500,
    /** Intro duration in ms before scroll takes over */
    introDuration: 6200,
    /** Star count desktop / mobile (<768px) */
    starCount: 280,
    starCountMobile: 150,
    /** Focal length for perspective projection */
    focalLength: 300,
    /** Phase thresholds (progress 0→1) */
    phases: {
      starsAppear: [0, 0] as const,
      warpAccel: [0.2, 0.6] as const,
      bhFade: [0.25, 0.7] as const,
      fullWarp: [0.55, 0.75] as const,
      flash: [0.58, 0.72] as const,
      reveal: [0.72, 0.92] as const,
      settled: 0.92,
    },
    /** Card stagger offset per index */
    cardStagger: 0.015,
  },
} as const;
