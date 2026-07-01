<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
at specs/005-demo-clip-download/plan.md
<!-- SPECKIT END -->

## Repository Conventions

- Package by feature. Put feature-owned code in the package that owns the
  product surface: `packages/aitraf-api` for serving/API behavior,
  `packages/aitraf-core` for reusable runtime processing, and
  `packages/aitraf-train` for offline data, training, evaluation, metrics,
  tracking, Hydra configs, and scripts. Only promote code to shared packages
  when multiple feature surfaces genuinely need it.
- Prefer functional programming. Keep business logic transformation-oriented
  with explicit inputs and outputs, small pure helpers where practical, and
  localized framework state at the boundary. Avoid hidden mutable state,
  cross-module side effects, and broad procedural flows.
- Prefer simple, modular code that is easy to extend. Implement the most basic
  working version first and add complexity only when the task requires it.
- Keep comments minimal and useful. Add them only when the code is not obvious.
- Read the existing project context and the specific file before editing it.

## Failure Behavior

- This is a training/evaluation pipeline. Break loudly when required inputs or
  artifacts are missing.
- Do not add fallback/default values to hide issues. If a required
  parameter/value is missing, raise an explicit error or ask for input.
- Do not silently fix, guess, patch, or mask unknown behavior.

## Agent Surfaces

- `AGENTS.md` is the single durable repo instruction file. Do not add a
  lowercase `agents.md`.
- `.agents/skills/` contains repo-local reusable agent workflows. Keep
  project-specific skills there rather than under `.codex/skills/`.
- `.specify/` contains Spec Kit templates, scripts, hooks, manifests, and
  generated integration state. Treat it as tool-owned infrastructure unless
  changing Spec Kit behavior intentionally.
- `specs/*` contains feature specs, plans, contracts, and task lists. The
  managed Spec Kit block above points to the active plan.
