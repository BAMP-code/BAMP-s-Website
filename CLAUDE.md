# BAMP's Website

Bryan's personal portfolio — a dark, **Cyberpunk 2077 / Edgerunners–themed**
showcase of projects, bio, and contact info. Deployed on **Vercel** at
bamp.codes.

> **Current design direction lives in [`docs/DESIGN_DIRECTION.md`](docs/DESIGN_DIRECTION.md)**
> ("Protocol // Krome" — a flat, wodniack.dev-inspired editorial page).
> That doc is the source of truth for look & feel; this file covers the
> engineering setup. The earlier 3D city-descent concept
> (`docs/REFACTOR_VISION.md`) was retired 2026-07-21.

## Tech Stack

- **Monorepo**: pnpm v9 + Turborepo v2
- **Framework**: Astro 5 (static output) + TypeScript 5.4 (strict). No UI
  framework — pages are `.astro` with small inline `<script>` islands.
- **Project data**: Supabase (Postgres), fetched at build time and
  Zod-validated (`src/lib/projects.ts`) — see "Content Architecture" below.
- **Styling**: Tailwind CSS 3.4 + PostCSS + Autoprefixer
- **Smooth scroll**: Lenis (`src/lib/scroll.ts`), disabled under
  `prefers-reduced-motion`
- **Linting / format**: ESLint (shared `packages/eslint-config`) + Prettier
  (with `prettier-plugin-tailwindcss`)
- **Testing**: Vitest / Playwright scaffolding at the workspace root
  (`turbo test` / `test:e2e`); the web app has no suites yet
- **Node**: v20

## Workspace Layout

```text
apps/web/               → Astro portfolio app (all UI, pages, content)
packages/eslint-config/ → Shared ESLint config
packages/tsconfig/      → Shared TS configs (base, nextjs, node)
docs/DESIGN_DIRECTION.md → Current visual direction (source of truth)
docs/brand/             → Brand identity documentation
```

`apps/web/src/`:
```text
components/site/  → page sections (SiteNav, Hero, Marquee, WorkList,
                    AboutSection, ContactSection, SiteFooter, BinaryRule)
content/          → about.ts, nav.ts (typed content modules; projects
                    live in Supabase — see Content Architecture)
layouts/Base.astro → <head>, fonts, HUD frame, smooth-scroll bootstrap
lib/              → scroll.ts (Lenis), types.ts, supabase.ts (client),
                    projects.ts (Zod-validated data access)
pages/index.astro → composes the single-page site
styles/global.css → tokens + all cyberpunk effects (scanlines, glitch,
                    marquee, work-row, corner brackets)
supabase/schema.sql → table schema, RLS policies, seed data — run once
                    in the Supabase SQL editor
```

## Common Commands

```bash
pnpm dev          # Start dev server (localhost:3000)
pnpm build        # Production build (astro build)
pnpm typecheck    # astro check
pnpm lint         # ESLint across workspace
pnpm test         # Vitest (via turbo)
pnpm format       # Prettier across the repo
```

## CI Pipeline (GitHub Actions)

`.github/workflows/ci.yml` runs on push to `main` and all PRs:
lint → typecheck → test → build.

## Design System

Full rationale, palette, and type in **`docs/DESIGN_DIRECTION.md`**.
Tokens are defined in `apps/web/tailwind.config.mjs`; font-family CSS
vars in `apps/web/src/styles/global.css`.

### Fonts
- **Anton** — ultra-bold condensed display caps (`--font-display`,
  Tailwind `font-display`). Hero, section + work titles, big numbers.
- **Share Tech Mono** — all HUD/mono (`--font-mono`, `font-mono`): nav,
  labels, status, binary rules, hash IDs, marquee, stat captions.
- **Rajdhani** — techno body/UI sans (`--font-body`, `font-body`/`sans`).
- Loaded via Google Fonts in `layouts/Base.astro`.

### Color Palette (Cyberpunk 2077 / Edgerunners)

| Token                | Hex       | Usage                              |
|----------------------|-----------|------------------------------------|
| `surface`            | `#0a0a07` | Warm-black background              |
| `surface-alt`        | `#111009` | Elevated panels / hover rows       |
| `border`             | `#23231a` | Warm hairline rules                |
| `hazard` / `brand-core` | `#fcee0a` | **Primary accent — 2077 yellow** |
| `accent`             | `#00f0ff` | Cyan (status, secondary)           |
| `accent-secondary`   | `#ff003c` | Alert red (glitch channel)         |
| `signal`             | `#00ff9f` | Signal green (available / go)      |
| `primary`            | `#ece9d8` | Warm off-white body text           |
| `muted`              | `#86866e` | Dim warm-grey labels               |
| `card-title`         | `#ffffff` | Headings                           |
| `card-body`          | `#cfcdba` | Body copy                          |
| `section-title-start`| `#fcee0a` | Heading gradient start (gold)      |
| `section-title-end`  | `#00f0ff` | Heading gradient end (cyan)        |

Yellow is the star and appears the most; cyan/green/red are one-job spices.

### Brand Identity
- **`BAMP//`** wordmark + a yellow **✦** spark (hero + favicon).
- Loud skin, calm spine: 2077 HUD framing over restrained editorial layout.
- Legacy brand docs in `docs/brand/`; `docs/DESIGN_DIRECTION.md` supersedes
  them for the site itself.

### Breakpoints
- `xs`: 400px, `sm`: 600px, `md`: 800px, `lg`: 992px

### Border Radius
- `card`: 20px, `card-lg`: 28px, `chat`: 16px, `pill`: 9999px

## Content Architecture

**Projects and categories live in Supabase**, not in the codebase —
`apps/web/src/lib/projects.ts` fetches them at build time via
`apps/web/src/lib/supabase.ts` and validates every row with Zod
(`supabase/schema.sql` has the table definitions, RLS policies, and
seed data). Categories are just rows in a `categories` table — add one
in the Supabase dashboard and it shows up as a work-list filter pill
with zero code changes. Requires `SUPABASE_URL` / `SUPABASE_ANON_KEY`
in `apps/web/.env` (see `.env.example`) and in Vercel's project env
vars; the build fails loudly if they're missing or a row doesn't match
the expected shape — there's no hardcoded fallback data.

Everything else still lives in `apps/web/src/content/` as typed
TypeScript modules:
- `about.ts` — bio, socials, portrait, skills, education, résumé
- `nav.ts` — navigation links

Types defined in `apps/web/src/lib/types.ts` (project/category types
are inferred from the Zod schema in `lib/projects.ts` instead).

## Key Conventions

- **Accessibility first**: skip link, visible focus, real alt text,
  keyboard-operable work list (`<details>`), `prefers-reduced-motion`
  honored for every animation.
- **Work list**: each project is a `<details>` row that animates open
  (Web Animations API in `WorkList.astro`); media sits in a uniform
  fixed-ratio frame so dropdowns are consistent regardless of image size.
  Category filter pills above the list are generated from whatever
  categories exist in Supabase and filter client-side (all project data
  is already baked into the page at build time — no re-fetch on filter).
- **Images**: plain `<img loading="lazy">` with `width`/`height` set;
  keep those attributes accurate to reserve layout space.
- **Effects**: CSS-driven (scanlines, glitch, marquee), all guarded by
  `prefers-reduced-motion` and toggleable via the nav **FX** switch.
- **Branch workflow**: feature branches → PR → CI passes → merge to `main`.
