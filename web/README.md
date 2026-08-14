# Otari admin dashboard

A React + [HeroUI v3](https://www.heroui.com) single-page app for the Otari
gateway's standalone admin panel: browse the model catalog, set model pricing,
manage aliases, and toggle runtime settings (model discovery and default
pricing). It talks to the gateway's management API (`/v1/models`, `/v1/pricing`,
`/v1/aliases`, `/v1/settings`) using the master key, which the operator enters
on the sign-in screen and which is held only in the browser tab's session
storage.

## Stack

- React 19 + TypeScript, built with Vite
- HeroUI v3 (`@heroui/react`, `@heroui/styles`) on Tailwind CSS v4
- TanStack Query for data fetching
- Vitest + Testing Library for tests

## Develop

```bash
cd web
npm install
npm run dev        # Vite dev server on :5173, proxying the API to :8000
npm run typecheck
npm test
```

`npm run dev` serves only the SPA, so it proxies `/v1` and `/health` to a
gateway at `http://localhost:8000` (see `vite.config.ts`). Start one first, for
example `uv run otari serve --config config.yml`, then sign in with that
gateway's master key. To develop against a gateway running elsewhere:

```bash
OTARI_DEV_API=https://your-app.up.railway.app npm run dev
```

If the source is edited through a bind mount (an agent working in a container,
say) and hot reload misses changes, the host watcher may not see the writes as
filesystem events. Fall back to polling:

```bash
VITE_USE_POLLING=1 npm run dev
```

## Build

```bash
npm run build
```

`npm run build` writes the production bundle to `../src/gateway/static/dashboard`
(configured in `vite.config.ts`). That directory is gitignored, not committed:
Vite content-hashes every asset filename, so a committed bundle made any two
branches touching `web/src` conflict on every file. There is nothing to commit
after a rebuild.

Who builds it instead:

- The Docker image builds it in a `node:22-slim` stage (see `Dockerfile`), so a
  container ships the dashboard with no action from you.
- From a source checkout, run `make dashboard` (repo root) once. Without a built
  bundle the gateway serves the get-started tutorial at `/` instead.
- A wheel ships the dashboard only if the bundle was built before
  `uv build`; `make dashboard` first if you need one that does.

CI (`.github/workflows/otari-dashboard.yml`) type-checks, tests, and builds on
every change under `web/`.

## How it is served

In standalone mode the gateway serves `index.html` at `/` and the hashed assets
under `/assets` (see `src/gateway/main.py` and `src/gateway/dashboard.py`). The
app uses client-side section switching rather than a router, so no server-side
catch-all route is needed. In hybrid mode there is no local management API, so
the root keeps serving the get-started tutorial instead.
