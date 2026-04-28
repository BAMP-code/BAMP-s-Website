# Performance audit — bamp.codes

Audit dated 2026-04-27. Reference benchmarks: wodniack.dev, animejs.com, logartis.info.

## TL;DR

The slowness is in the payload and the animation pipeline, not the host. Vercel's edge is already faster than anything self-managed; switching to Kubernetes/GCP would be a regression in both latency and ops cost. Three root causes, ranked by pain:

1. ~75 MB of public assets (one 50 MB video, one 21 MB headshot)
2. Two competing wheel-event hijackers fighting native scroll
3. SVG turbulence + displacement + blur filter chain animated every frame, in two places

## 1. Asset weight

```
apps/web/public/videos/link-app-demo.mp4    50 MB
apps/web/public/images/headshot.JPG         21 MB
apps/web/public/images/Ghost.jpg           839 KB
apps/web/public/images/Cook.jpg            724 KB
apps/web/public/images/Useless_box.jpg     723 KB
apps/web/public/images/LED_board.jpg       553 KB
apps/web/public/images/Truss_project.jpg   437 KB
```

The 21 MB headshot is the most surprising find. Vercel's image optimizer has a 4 MB source ceiling; past that it serves the original. Ship a 1600 px master JPEG (~250 KB) and let `next/image` regenerate AVIF/WebP variants from there.

The 50 MB video should be re-encoded H.264 1080p ~3 Mbps (~3 MB for a 10 s clip). Two renditions (720p + 1080p) via `<source media>` is even better. Also change `preload="auto"` to `preload="metadata"` at `apps/web/components/projects-timeline.tsx:132` so the modal doesn't fetch the whole file on mount.

macOS Cloud-sync duplicates need to be deleted, not committed:

- `apps/web/components/image-preloader 2.tsx`
- `apps/web/components/social-icons 2.tsx`
- `apps/web/components/chatbot 2/`
- `apps/web/components/slider 2/`
- `apps/web/lib/data 2/`
- `apps/web/app/api 2/`
- `apps/web/public/images 2/`
- `apps/web/public/videos 2/`
- `apps/web/public/resume 2.pdf`
- `.DS_Store` in `.claude/`, `.github/`, `apps/`, `packages/`, repo root
- `apps/web/tsconfig.tsbuildinfo` (should be gitignored)

## 2. Scroll feels off — three handlers competing

### a. `html { scroll-behavior: smooth }` at `apps/web/app/globals.css:12`

CSS-driven smooth scroll for anchor jumps. Fights any JS-driven scroll. Modern smooth-scroll libraries (Lenis, GSAP ScrollSmoother) require this to be off.

### b. `WarpSection` wheel hijack at `apps/web/components/warp-section.tsx:70-95`

Calls `e.preventDefault()` on every wheel event and re-applies `scrollBy` at 35–75% speed. The "broken scroll" antipattern:

- Trackpad inertia gets multiplied by 0.35 → smooth decay becomes stutter.
- Touch input doesn't fire `wheel` events, so iPad/iPhone Safari users scroll at full speed while desktop users scroll dampened — the animation timing diverges between input methods.
- Blocks keyboard scroll (PageDown, Space) and assistive tech.
- The IntersectionObserver-style early-return doesn't actually disable the listener — it stays attached on first paint.

### c. `ProjectsTimeline` wheel hijack at `apps/web/components/projects-timeline.tsx:316-336`

A second `preventDefault` wheel listener that translates vertical wheel into horizontal `translate3d`. Combined with (b), the user has different scroll physics in different sections of one page. That mismatch is exactly why scroll feels wrong.

### What the reference sites do

- **wodniack.dev** uses GSAP ScrollTrigger (visible from their "GSAP SOTM" awards). ScrollTrigger does **not** preventDefault — it observes scroll and computes progress from `getBoundingClientRect`. Same approach you're already using for progress, just without the wheel hijack.
- **animejs.com** uses anime.js's ScrollObserver, also non-preventing.
- **logartis.info** uses native scroll throughout.

Recommendation: rip out both wheel hijackers. Keep the `getBoundingClientRect`-based RAF for progress. Optionally add Lenis (~6 KB) for buttery momentum if you want the wodniack feel.

## 3. SVG filter chain — most expensive way to draw a black hole

Both `BlackHoleHero` and `BlackHoleFooter` mount this filter:

```
feTurbulence → feDisplacementMap → feGaussianBlur → feMerge
```

…and animate `baseFrequency`, `scale`, and `stdDeviation` every RAF tick.

- `feTurbulence` regenerates noise on every attribute change.
- `feDisplacementMap` re-samples the source image per pixel.
- Animating `stdDeviation` re-runs the blur shader.
- Firefox is dramatically worse here. iOS Safari drops frames hard.
- Both hero and footer run separate RAFs doing this.

### Concurrent RAF loops on first paint

1. `WarpSection` scroll-progress RAF
2. `StarfieldCanvas` 280-star draw RAF
3. `BlackHoleHero` mouse-reactive SVG-filter RAF
4. `BlackHoleHero` letter-suck RAF (intro)
5. `BlackHoleFooter` mouse-reactive SVG-filter RAF
6. `ProjectsTimeline` wheel-scroll RAF
7. `VideoModal` scrubber RAF (if open)

Up to 5–7 simultaneous RAFs. Each frame, all of them do work.

### Fixes

- **Pre-bake `feTurbulence`** as a static `<feImage>`. The visual delta of `baseFrequency` changes of 0.0015 per frame is imperceptible; you're paying for re-rendering noise that looks identical.
- **Replace SVG with a fragment shader** for the hero black hole. ~80 lines of GLSL, GPU-bound, ~3 KB OGL or twgl. This is the single biggest "premium feel" upgrade.
- **Gate footer RAF on visibility properly.** `apps/web/components/black-hole-footer.tsx:24` already has an IntersectionObserver, but the RAF still spins and just early-returns at line 84. Move IO to start/stop the loop.
- **`BlackHoleHero` doesn't gate on visibility at all.** It runs until `progress >= 1` — burning frames in parallel with the warp animation.

## 4. Other cuts

- **Layout loads two Google fonts**: Manrope + DM_Mono with three weights. Drop DM_Mono unless actually used in `<code>` blocks. Reduce Manrope weights.
- **No `next/dynamic` splitting.** `VideoModal`, chatbot, footer all ship in initial JS.
- **Static export.** Homepage has no SSR-required logic. Add `export const dynamic = "force-static"` on `apps/web/app/page.tsx`.
- **`tsconfig.tsbuildinfo` is committed.** Add to `.gitignore`.
- **Sentry** adds ~30 KB gzipped for a portfolio site that likely doesn't need error monitoring.
- **`priority` on timeline images** at `projects-timeline.tsx:403, 460` — these images are below the warp section, so they're not LCP candidates. `priority` here front-loads bytes that aren't urgent.

## Why not Kubernetes on GCP

- Vercel Edge runs in 140+ POPs already, with automatic image optimization, ISR, and request collocation. K8s on GCP would add 20–50 ms cold-start latency in regions Vercel covers and you don't.
- Static-leaning Next.js sites cost ~$0/mo on Vercel hobby. K8s adds compute, networking, and observability bills.
- Operational burden: cluster upgrades, image hardening, certificate rotation, secret management. None of which you need.
- The reference sites you admire are static. wodniack is Astro static, animejs is doc-site static, logartis is Angular SPA with static assets. Each wins by being lean, not by infrastructure.

## Implementation plan

### Tier 1 — fixes 80% of perceived slowness

1. Re-encode `headshot.JPG` to ~1600 px / ~250 KB.
2. Re-encode `link-app-demo.mp4` H.264 1080p ~3 Mbps; change `preload="auto"` → `preload="metadata"`.
3. Delete macOS-sync duplicates and `.DS_Store`. Add `.gitignore` entries.
4. Remove both wheel hijackers; remove `html { scroll-behavior: smooth }`.

### Tier 2 — premium polish month

5. Add Lenis (`npm i lenis`) and pipe both progress refs through one tick.
6. Pre-bake `feTurbulence`; animate only `feDisplacementMap.scale`.
7. Move both black-hole RAFs behind IntersectionObserver start/stop.
8. `next/dynamic` for `VideoModal`, chatbot, footer.
9. Drop `DM_Mono`; reduce `Manrope` weights.

### Tier 3 — wodniack-grade

10. Replace SVG black hole with fragment shader.
11. Add `animation-timeline: view()` for scroll-linked reveals where supported (progressive enhancement).
12. `export const dynamic = "force-static"` on homepage.
13. `prefers-reduced-data` branch with static black-hole PNG fallback.

## Reference-site teardown

| Site | Stack | Smooth scroll | Animation | Trick |
|---|---|---|---|---|
| wodniack.dev | Astro static | GSAP ScrollTrigger | GSAP timelines | Astro ships zero JS by default; only animation islands hydrate. AVIF + hashed filenames via Astro's image pipeline. |
| animejs.com | Static + anime.js v4 | Native + their ScrollObserver | Anime.js timelines + WAAPI | Modular import keeps bundle at 24.5 KB; one shared RAF. |
| logartis.info | Angular SPA | Native | Custom CSS + Angular animations | Heavy initial JS but content/images fully static; client-side state after first load. |

Common thread: small JS, tight critical path, GPU-accelerated motion.
