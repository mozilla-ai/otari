# Web dashboard (`web/`)

`web/CLAUDE.md` is a one-line `@AGENTS.md` import. Edit this file and never
replace or remove the import.

Before changing the dashboard, read
[frontend-standards](../.github/skills/frontend-standards/SKILL.md) and the one
topic guide it points to for the work at hand.

## Runtime contract

`src/main.tsx` fetches unauthenticated `GET /v1/bootstrap` before mounting
React. Do not guess a deployment when that request fails.

The bootstrap selects standalone, hosted, or hybrid presentation and publishes:

- sign-in methods and configured OAuth providers
- management surfaces hosted by this process
- the control-plane or data-plane URL
- documentation, mail, passkey, and maintenance state

Features read this through `useDeployment()` and `useSurfaces()`. Page
components do not branch directly on the mode. A surface gate controls
discoverability; backend authorization remains mandatory.

Local sign-in exchanges a master key, password, passkey, or OAuth assertion for
an HttpOnly session cookie. Do not store credentials or copy the cookie into
JavaScript state.

Public auth and invitation flows render ahead of the authenticated router in
`DeploymentRoot`. Their URLs are backend contracts used by email and OAuth
redirects, so changing one requires changing the producer and its tests.

Request snippets use `shared/helpers/requestSnippets.ts`. Standalone and
hybrid use the browser origin; hosted uses the bootstrap's `data_plane_url`.
A missing hosted data-plane URL renders no runnable snippet.

## Source layers

Biome enforces these dependencies:

| Layer | May import |
| --- | --- |
| `shared/` | `client/` and itself |
| `features/` | `shared/`, `client/`, and other features |
| `app/` | any dashboard layer |
| `routes/` | feature pages and shared composition |
| `tests/` | any layer |

Do not import an overlay tree from the base dashboard. Otari.ai composes its UI
through the explicit overlay seams.

## Navigation

`src/app/nav/registry.ts` owns sidebar navigation. Items can be gated by:

- `surface`, the deployment topology
- `capability`, the installed or entitled feature
- `operatorOnly`, deployment-wide authority

These gates combine with AND. Organization and workspace authorization still
comes from server responses.

The workspace and organization rails have separate registries. A sidebar item
points to a real page, never a redirect. `/docs` and `/account` are chrome
destinations and do not belong in a rail.

Overlay navigation uses the empty seam modules under `src/app/nav/` for new
sections, items inserted into base sections, and label overrides. Keep their
types aligned with otari-ai.

## Data and generated files

TanStack Query owns server state. Query keys, mutation invalidation, pagination,
and loading behavior follow the frontend standards topic guides.

Import API shapes from `@/client`, not directly from the generated schema.
`src/client/schema.ts` is generated from `docs/public/openapi.json` and
committed. Keep `src/client/local.ts` limited to shapes OpenAPI cannot own.

File routes live in `src/routes/`. Each route file exports `Route` and
nothing else so automatic code splitting works. The generated
`src/routeTree.gen.ts` is committed.

Search parameters use the shared flat repeated-key codec and `useUrlState`.
Component tests for URL-aware pages use the real router helpers and await its
first resolution.

## Toolchain and build

Use pnpm as pinned in `package.json`. The Docker and CI Node major is 22, so
`@types/node` follows 22. Keep shared React Aria dependencies on the version
HeroUI resolves, and approve required install scripts in
`pnpm-workspace.yaml`.

The React Compiler runs through the configured Babel pass. Avoid reflexive
`useMemo`, `useCallback`, and `React.memo`; effects still need correct
dependencies and cleanup.

`make dashboard` builds to `src/gateway/static/dashboard/`. The bundle is
gitignored and not committed. Docker builds it. A source checkout without it
serves the welcome page.

Two build inputs are committed when changed:

- `web/src/client/schema.ts`
- `web/src/routeTree.gen.ts`

The bundled dashboard guide imports `docs/dashboard.md`. It remains available
at `/#/docs` even when `docs_url` points the visible links elsewhere.

The PWA manifest is generated at build time by `web/pwaManifest.ts`, which
prefixes `id`, `start_url`, `scope`, and each icon path with Vite's resolved
`base` and rewrites the `index.html` link (#857). Do not add a static manifest
to `public/pwa/`; only the icons live there. The plugin is a standalone module
so the otari-ai superset build imports it rather than copying it.

## Checks

```bash
pnpm --dir web run lint
pnpm --dir web run typecheck
pnpm --dir web test
pnpm --dir web run build
```

Playwright behavioral tests run against a real built gateway and scope
assertions to the rows they create. Dismiss React Aria popovers before asserting
outside them.

Screenshot tests cover three viewports and both themes. They are
workflow-dispatch only, their baselines are gitignored, and CI artifacts are the
review output until the suite becomes a pull-request gate.
