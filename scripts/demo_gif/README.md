# Dashboard demo GIF

Regenerates `assets/otari-demo.gif` (the README hero): a full sweep of the
standalone admin dashboard populated with realistic providers, users, budgets,
keys, and ~1600 usage rows.

## Regenerate

```bash
bash scripts/demo_gif/record.sh
```

That seeds a throwaway SQLite gateway, boots it serving the committed dashboard
bundle, drives every page with Playwright while recording, and encodes the
result to `assets/otari-demo.gif`.

## Prereqs

- `ffmpeg` on `PATH` (webm to gif).
- `gifsicle` on `PATH` (lossy GIF optimisation).
- Web deps installed: `(cd web && npm ci)` (Playwright resolves from
  `web/node_modules`).
- A committed dashboard bundle in `src/gateway/static/dashboard`. Rebuild after
  any `web/src` change: `npm --prefix web run build`.
- On arm64 Linux, `record.sh` sets the Playwright platform override automatically; on x86_64 it leaves Playwright's native detection and host checks in place.

## Files

| File | Purpose |
| --- | --- |
| `otari.yml` | Gateway config (SQLite, fixed master key `otari-demo-key`). |
| `seed.py` | Deterministic seed: providers, budgets, users, keys, pricing, aliases, usage. |
| `tour.mjs` | Playwright script: signs in and sweeps the pages, recording video + per-page screenshots. |
| `record.sh` | Orchestrates seed to boot to record to encode. |

## Tweaking

- **Data**: edit the tables at the top of `seed.py` (providers, budgets, users,
  keys, models, aliases). RNG is seeded (`4242`) so runs are reproducible.
- **Pace / route order / dwell**: edit `STOPS` in `tour.mjs`.
- **Size / fps / quality**: edit `FPS`, `WIDTH`, and the palette flags in
  `record.sh`.

Intermediate output (recording, per-page screenshots, palette, server log)
lands in `scripts/demo_gif/artifacts/` (gitignored) for inspection.
