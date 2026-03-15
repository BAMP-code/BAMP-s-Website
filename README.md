# BAMP Website Monorepo

Personal portfolio and chatbot stack, organized as a pnpm + Turborepo workspace.

## Structure
- `apps/web`: Next.js portfolio frontend (App Router, TypeScript, Tailwind)
- `packages/eslint-config`: Shared lint config
- `packages/tsconfig`: Shared TypeScript config
- `docs/brand`: Brand system and symbol concept notes

## Commands
- `pnpm dev`: Run all app dev tasks
- `pnpm lint`: Run workspace lint checks
- `pnpm typecheck`: Run workspace TypeScript checks
- `pnpm test`: Run workspace tests
- `pnpm build`: Build workspace projects

## CI
GitHub Actions workflow at `.github/workflows/ci.yml` runs lint, typecheck, test, and build on push/PR.

## Brand Assets
- Selected living mark: event-horizon symbol
- Assets are in `apps/web/public/brand`
