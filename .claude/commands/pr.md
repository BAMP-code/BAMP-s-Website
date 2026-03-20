---
description: Create a pull request with quality checks
---

You are about to create a PR for the user's current branch. Follow these steps:

## Step 1: Pre-flight

Run in parallel:
- `git branch --show-current` — must NOT be `main`
- `git status` — check for uncommitted changes
- `pnpm typecheck` — must pass
- `pnpm lint` — must pass

If on `main`, stop and tell the user to create a feature branch.
If there are uncommitted changes, ask if they should be committed first (use `/commit` flow).
If checks fail, show errors and offer to fix before proceeding.

## Step 2: Analyze Changes

Run `git log main..HEAD --oneline` and `git diff main...HEAD --stat` to understand all changes in this branch.

## Step 3: Create PR

- Push the branch with `git push -u origin HEAD`
- Draft a clear PR title (under 70 chars) and body
- Use this format for the body:

```
## Summary
<2-4 bullet points of what changed and why>

## Test plan
- [ ] CI passes (lint, typecheck, tests, build)
- [ ] <specific manual checks relevant to the changes>
```

- Show the draft to the user for approval, then create with `gh pr create`
- Return the PR URL when done
