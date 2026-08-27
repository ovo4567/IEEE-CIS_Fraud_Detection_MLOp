# 11: Docs + HANDOFF polish

**What to build:** The portfolio-facing documentation — a README with an architecture diagram and a `make demo` guide, plus the `HANDOFF.md` deployment section updated to reflect the new state of the project.

**Blocked by:** 09: Docker Compose + make demo

**Status:** ready-for-agent → in-progress → resolved

- [x] README has an architecture diagram and a `make demo` quickstart
- [x] `HANDOFF.md` deployment section reflects the current project state
- [x] Portfolio notes capture the MLOps story for reviewers

## Answer

### What was built

Docs only — no code changed (the suite stays 151 passing). The
portfolio-facing documentation is now real:

- **`README.md`** — rewritten from the cookiecutter boilerplate into the
  portfolio-facing entry point: a one-paragraph pitch, key numbers (test AUC
  0.9286, operating threshold 0.0551), a **Mermaid architecture diagram** of the
  closed loop (train → serve → monitor → retrain → promote, with the shared
  model volume and "no redeploy on promotion" shown), a **`make demo`
  quickstart** (prerequisites + a service/URL table + `demo-logs`/`demo-down`),
  the one-command loop (`make simulate`/`monitor`/`retrain`), the CI/CD
  summary, a repo layout that reflects the real package (`serving`,
  `orchestration`, `monitoring`, `deployment`), the quality gates, the ADR
  index, and doc links.
- **`docs/docs/portfolio.md`** — the **MLOps story for reviewers**: the problem,
  the model, the four ADR decisions, the three-phase closed loop, why it's a
  credible deployment (fresh-clone verification, honest label arrival, real
  DeLong gate, CI gating), exactly what a reviewer should run, and honest
  limitations + v2 ideas.
- **MkDocs** — `docs/mkdocs.yml` gains a nav (Home / Getting started /
  Portfolio); `docs/docs/index.md` is now a landing page linking the portfolio,
  getting-started, ADRs, and CONTEXT.md.
- **`HANDOFF.md`** — header updated (ticket 11 done), the Docker Compose demo
  added to the Current state table (it was missing), and a ticket 11 session
  section appended.

### Verification

- All doc links resolve to existing files (`deploy/README.md`,
  `docs/docs/portfolio.md`, `getting-started.md`, `CONTEXT.md`, `docs/adr/`).
- No Python/package code touched → `make test` / `make lint` unaffected by
  this ticket; the full suite is re-run as the standard close-out gate.
- Mermaid blocks are `flowchart` (GitHub + MkDocs render natively).

### Design notes

- Kept the README scannable (pitch + diagram + quickstart) and pushed the
  longer narrative to `docs/docs/portfolio.md` — reviewers get the story on the
  landing page, operators get the how-to in `getting-started`/`deploy/README.md`.
- The architecture diagram makes the *loop* explicit (stream replay → scoring →
  monitoring → retrain → promote → served without redeploy) rather than just a
  box diagram, since the closed loop is the project's thesis.
