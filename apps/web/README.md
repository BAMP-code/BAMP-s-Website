# @bamp/web

Next.js portfolio application for BAMP.

## Tech
- Next.js (App Router) + TypeScript
- Tailwind CSS
- Vitest + Testing Library
- Playwright (e2e)

## Key Paths
- `app/page.tsx`: Page composition
- `app/layout.tsx`: Metadata, analytics, app shell
- `app/globals.css`: Global styles and design tokens
- `components/*`: Reusable UI sections
- `content/*`: Editable portfolio content
- `lib/motion.ts`: Shared motion timing and thresholds
- `public/brand/*`: Symbol, favicon, and social card assets

## Local Commands
- `pnpm -C apps/web dev`
- `pnpm -C apps/web lint`
- `pnpm -C apps/web typecheck`
- `pnpm -C apps/web test`
- `pnpm -C apps/web build`

## Editing Workflow
1. Update content in `content/` for text/media changes.
2. Update look/feel through `globals.css`, `tailwind.config.ts`, and components.
3. Keep motion values centralized in `lib/motion.ts`.
4. Run lint + typecheck + tests before shipping.
