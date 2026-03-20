---
description: Create a new feature branch from latest main
---

Create a new feature branch for the user. Follow these steps:

1. Ask the user what they're working on (if not already clear from context)
2. Run `git fetch origin main` to get latest
3. Create and switch to a new branch from `origin/main` using a descriptive name:
   - Format: `feat/<short-description>`, `fix/<short-description>`, or `refactor/<short-description>`
   - Use kebab-case, keep it short (2-4 words)
4. Confirm the new branch name and that it's based on the latest `main`
