# Otari admin dashboard

A React + [HeroUI v3](https://www.heroui.com) single-page app for the Otari
gateway's standalone admin panel: create and revoke virtual API keys, manage
users, and observe usage and traffic. It talks to the gateway's existing
management API (`/v1/keys`, `/v1/users`, `/v1/usage`) using the master key,
which the operator enters on the sign-in screen and which is held only in the
browser tab's session storage.

## Stack

- React 19 + TypeScript, built with Vite
- HeroUI v3 (`@heroui/react`, `@heroui/styles`) on Tailwind CSS v4
- TanStack Query for data fetching, Recharts for usage charts
- Vitest + Testing Library for tests

## Develop

```bash
cd web
npm install
npm run dev        # Vite dev server (proxy/point it at a running gateway)
npm run typecheck
npm test
```

## Build

```bash
npm run build
```

`npm run build` writes the production bundle to `../src/gateway/static/dashboard`
(configured in `vite.config.ts`). That directory is committed to the repository
so the Python package and Docker image ship the dashboard without a Node build
step. After changing anything under `web/src`, rebuild and commit the updated
bundle. CI (`.github/workflows/otari-dashboard.yml`) fails if the committed
bundle is stale.

## How it is served

In standalone mode the gateway serves `index.html` at `/` and the hashed assets
under `/assets` (see `src/gateway/main.py` and `src/gateway/dashboard.py`). The
app uses client-side section switching rather than a router, so no server-side
catch-all route is needed. In hybrid mode there is no local management API, so
the root keeps serving the get-started tutorial instead.
