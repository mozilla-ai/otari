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
- TanStack Query for data fetching, TanStack Router (file-based) for routes
- Vitest + Testing Library for tests
- Biome for formatting and lint, including the layer boundary (`npm run lint`, `npm run lint:fix`)

## Layout

`src/` is four layers plus a test corner:

| Directory | Holds | May import |
| --- | --- | --- |
| `features/<domain>/` | A domain's page, the parts only it uses, its tests | `shared/`, `client/`, other features |
| `shared/` | `components/` primitives, `helpers/` pure functions, `api/` transport and query hooks, `hooks/` cross-cutting React state | `client/`, itself |
| `app/` | The composition root: providers, router, shell chrome, and `nav/`, the sidebar registry and its overlay seam | any layer here |
| `routes/` | One file per URL, naming a feature's page | any layer here |
| `tests/` | Test harnesses; outside the boundary, so a harness may mount the app's providers | any layer here |

`client/` is generated from the OpenAPI spec, `main.tsx` is the entry, and
`styles/globals.css` is the one stylesheet. This is `otari-ai/frontend/src`'s
layout, because that control-plane UI moves into this repo at M5.

`npm run lint` fails a PR that has a feature importing `app/`, or `shared/`
importing either of the layers above it. No layer may import an overlay's tree,
which lives in `otari-ai` and is composed onto this base at build time.
`src/architecture.test.ts` proves the lint still rejects each of those. See
[AGENTS.md](./AGENTS.md) for the reasoning.

## Develop

```bash
cd web
npm install
npm run dev        # Vite dev server on :5173, proxying the API to :8000
npm run lint       # format + lint, including the layer boundary
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

CI (`.github/workflows/otari-dashboard.yml`) lints, type-checks, tests, and
builds on every change under `web/`.

## How it is served

In standalone mode the gateway serves `index.html` at `/` and the hashed assets
under `/assets` (see `src/gateway/main.py` and `src/gateway/dashboard.py`). The
app routes on hash history (`/#/models`, `/#/usage`), so no server-side
catch-all route is needed. In hybrid mode there is no local management API, so
the root keeps serving the get-started tutorial instead.
