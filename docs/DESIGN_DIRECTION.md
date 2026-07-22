# Design Direction — bamp.codes v2 · "Protocol // Krome"

Adopted 2026-07-20. This supersedes the 3D city-descent concept in
`REFACTOR_VISION.md` for the **public homepage**. The scene components
(`src/components/scene/*`) are retained on disk but no longer mounted.

## Reference

Structure and interaction language are modeled on **wodniack.dev**
(Antoine Wodniack — Awwwards SOTD / CSS Design Awards). What we borrow:

- One enormous typographic statement as the hero ("CREATIVE ✦ ENGINEER").
- **Binary code (0/1) as section dividers.**
- A **hash-ID work list** — indexed rows, monospace hash per item, that
  reveal on interaction (not a grid of cards).
- An availability status line + a small stat block.
- Animation-driven but restrained; a contrast/HUD sensibility.

We do **not** copy it — layout, copy, and skin are ours.

## Chosen skin — "Protocol // 2077"

The Cyberpunk 2077 main-menu attitude laid over the wodniack skeleton:
pure warm-black ground, hazard-yellow as the dominant accent, faint
scanlines, corner brackets framing the viewport, an ambient glitch pass
on the headline, one data-marquee ("hazard tape") used **once**. The
work list itself is the calmer "Krome" treatment (design 3): quiet
indexed rows that light up yellow on hover/expand.

Loud skin, calm spine. Spend the boldness on the hero and the marquee;
keep the work list and about section quiet.

## Palette — Cyberpunk 2077 / Edgerunners

| Token              | Hex       | Role                                   |
|--------------------|-----------|----------------------------------------|
| `surface`          | `#0a0a07` | Warm-black ground                      |
| `surface-alt`      | `#111009` | Elevated panels / hover rows           |
| `border`           | `#23231a` | Warm hairline rules                    |
| `hazard` / brand   | `#fcee0a` | **Primary accent — 2077 yellow**       |
| `accent`           | `#00f0ff` | Cyan (status, secondary highlights)    |
| `accent-secondary` | `#ff003c` | Alert red (glitch channel)             |
| `signal`           | `#00ff9f` | Signal green (available / go)          |
| `primary`          | `#ece9d8` | Warm off-white body text               |
| `muted`            | `#7a7a66` | Dim warm-grey labels                   |
| `card-title`       | `#ffffff` | Headings                               |

Yellow is the star and appears the most; cyan/green/red are spice —
one job each. Keep saturation high but let black dominate.

## Type

- **Display — Anton.** Ultra-bold condensed caps. Hero, section titles,
  work-row titles, big stat numbers. Brutalist poster energy.
- **Mono — Share Tech Mono.** All HUD: nav, labels, status, binary
  rules, hash IDs, the marquee, stat captions.
- **Body / UI — Rajdhani.** Techno squarish sans (the closest free
  match to the 2077 UI face) for bio copy and metadata. Weights 400–700.

Loaded via Google Fonts in `layouts/Base.astro`; family vars live in
`styles/global.css`; exposed to Tailwind as `font-display` / `font-mono`
/ `font-body`.

## Component map (`src/components/site/`)

- `SiteNav.astro` — `BAMP//` brand, anchor nav, contrast/HUD affordance.
- `Hero.astro` — status pill, glitch headline, meta line.
- `Marquee.astro` — the single data-ticker (design 1 styling).
- `WorkList.astro` — hash-ID rows from `content/projects.ts`; each row is
  a `<details>` that expands to description + media (no-JS accessible).
- `AboutSection.astro` — portrait, bio, education, skill chips.
- `ContactSection.astro` — "let's build something loud" + socials + résumé.
- `SiteFooter.astro` — binary rule + colophon.

Shared effects (scanlines, corner brackets, glitch, marquee keyframes,
work-row hover, status pulse) live in `styles/global.css`, all guarded
by `prefers-reduced-motion`.

## Non-negotiables

1. Cyberpunk, not Apple-minimal. Yellow-on-black, HUD framing.
2. wodniack structure: giant type, binary rules, hash-ID work list.
3. Accessible: `<details>` for expandable work, visible focus, reduced
   motion honored, real alt text.
4. One responsive layout (no separate desktop/mobile trees).
