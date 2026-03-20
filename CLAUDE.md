# BAMP's Website

Bryan's personal portfolio website — a dark, space-themed showcase of projects, bio, and contact info. Deployed on **Vercel** at bamp.codes.

## Tech Stack

- **Monorepo**: pnpm v9 + Turborepo v2
- **Framework**: Next.js 14 (App Router) + React 18 + TypeScript 5.4 (strict)
- **Styling**: Tailwind CSS 3.4 + PostCSS + Autoprefixer
- **Testing**: Vitest (unit), Playwright (E2E), @axe-core/playwright (a11y)
- **Linting**: ESLint (next/core-web-vitals, jsx-a11y), Prettier w/ Tailwind plugin
- **Monitoring**: Sentry, Vercel Analytics & Speed Insights
- **Node**: v20

## Workspace Layout

```
apps/web/          → Next.js portfolio app (all UI, pages, API routes)
packages/eslint-config/ → Shared ESLint config
packages/tsconfig/      → Shared TS configs (base, nextjs, node)
docs/brand/             → Brand identity documentation
```

## Common Commands

```bash
pnpm dev          # Start dev server (localhost:3000)
pnpm build        # Production build
pnpm lint         # ESLint across workspace
pnpm typecheck    # tsc --noEmit
pnpm test         # Vitest unit tests
pnpm test:e2e     # Playwright E2E tests
pnpm format       # Prettier format all files
```

## CI Pipeline (GitHub Actions)

Runs on push to `main` and all PRs: lint → typecheck → test → build.

## Design System

### Fonts
- **Manrope** (sans-serif) — primary body & headings (`--font-manrope`)
- **DM Mono** (monospace) — code & accent text, weights 300/400/500 (`--font-dm-mono`)

### Color Palette
| Token               | Hex       | Usage                        |
|----------------------|-----------|------------------------------|
| `primary`            | `#f4f6ff` | Primary text                 |
| `muted`              | `#9ba4c7` | Secondary/muted text         |
| `surface`            | `#060608` | Main background              |
| `surface-alt`        | `#0d0d12` | Elevated surfaces            |
| `border`             | `#28283c` | Borders & dividers           |
| `card-title`         | `#ffffff` | Card headings                |
| `card-body`          | `#d4d8ef` | Card body text               |
| `accent`             | `#00f6ff` | Cyan accent                  |
| `accent-secondary`   | `#ff2e2e` | Red accent                   |
| `brand-core`         | `#f6ea2a` | Yellow/gold brand color      |
| `section-title-start`| `#f6ea2a` | Heading gradient start (gold)|
| `section-title-end`  | `#00f6ff` | Heading gradient end (cyan)  |

### Brand Identity
- **Event Horizon Ring** symbol — offset ring + dense core + escape arc
- Dark base with electric accents (cyan/violet)
- Paired with "BAMP" wordmark in navigation
- See `docs/brand/symbol-system.md` for full usage rules

### Breakpoints
- `xs`: 400px, `sm`: 600px, `md`: 800px, `lg`: 992px

### Border Radius
- `card`: 20px, `card-lg`: 28px, `chat`: 16px, `pill`: 9999px

## Content Architecture

Content lives in `apps/web/content/` as TypeScript modules:
- `projects.ts` — project entries (categories: cs, ee-me, drawings)
- `about.ts` — bio, socials, portrait
- `nav.ts` — navigation links

Types defined in `apps/web/lib/types.ts`.

## Key Conventions

- **Accessibility first**: ARIA labels, focus traps, skip links, `prefers-reduced-motion` support
- **Lazy loading**: images and videos below fold are lazy-loaded with poster frames
- **Animation**: Intersection Observer for reveals, RAF for scroll-linked effects, CSS for continuous motion. Config in `apps/web/lib/motion.ts`
- **Image optimization**: Next.js `<Image>` with webp/avif formats
- **Branch workflow**: feature branches → PR → CI passes → merge to `main`
