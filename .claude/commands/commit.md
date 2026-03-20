---
description: Smart commit — validates branch, runs checks, then commits staged work
---

You are about to commit the user's work. Follow these steps carefully:

## Step 1: Branch Validation

Run `git branch --show-current` and `git status` (never use `-uall`).

- If the current branch is `main` or `master`, **STOP** and warn the user: "You're on `main`. Create a feature branch first." Do NOT commit to main.
- Look at the changed files and the branch name. If the changes don't seem related to the branch name, flag this to the user and ask if they want to continue or switch branches.

## Step 2: Quality Checks

Run these in parallel:
1. `pnpm typecheck` — TypeScript must pass with zero errors
2. `pnpm lint` — ESLint must pass

If either fails:
- Show the errors clearly
- Ask the user if they'd like you to fix them before committing
- Do NOT proceed with the commit until checks pass or the user explicitly says to skip

## Step 3: Review Changes

Run `git diff --staged` and `git diff` to see all changes. If nothing is staged, show the user what's unstaged and ask what they want to commit.

## Step 4: Commit

- **Split into logical commits**: Group related changes together into separate commits that each make sense on their own. For example, image compression should be its own commit, a new component should be its own commit, layout tweaks across files can be grouped, etc. Ask the user to confirm the proposed breakdown before committing.
- Stage the relevant files (prefer explicit file paths over `git add .`)
- Write a concise commit message that describes the **why**, not just the **what**
- Use conventional style: lowercase, imperative mood (e.g., "add contact section with social links")
- **Never** add "Co-Authored-By" lines to commit messages
- Present the commit message(s) to the user for approval before committing
- After committing, show the result with `git log --oneline` for all new commits
