---
description: Run all CI checks locally (lint, typecheck, tests)
---

Run the full CI quality gate locally, mirroring what GitHub Actions runs on PRs.

Run these in parallel:
1. `pnpm lint`
2. `pnpm typecheck`
3. `pnpm test`

Report results as a checklist:
- [ ] Lint — pass/fail (show errors if any)
- [ ] Typecheck — pass/fail (show errors if any)
- [ ] Tests — pass/fail (show failures if any)

If everything passes, confirm the branch is CI-ready.
If anything fails, summarize what needs fixing and offer to fix it.
