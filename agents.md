# Coding Agent Rules

- This is a training/evaluation pipeline. Break loudly. Errors are good.
- Never add fallback/default values to hide issues.
- If a required parameter/value is missing → raise error or ask for input.
- No silent fixes, no magic assumptions, no patching unknown behavior.
- Always read the existing project/codebase before doing anything.
- Always read the specific file before editing it.
- First step on any task: scan code → form a short plan → then implement.
- Keep responses concise. No filler, no storytelling.
- Prefer simplicity. Implement the most basic working version first. Add complexity only if asked.
- Prefer list comprehensions & higher-order functions over long/complex loops.
- Prefer immutable code
- Code must be modular, simple, clean, and easy to extend.
- Minimal comments. Only when something isn’t obvious from code.
- If unsure → ask instead of hallucinating.
