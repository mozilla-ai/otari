# Dashboard demo GIF

Regenerates `assets/otari-demo.gif` (the README hero): a tour of the standalone
dashboard's most useful pages, populated with seven providers, fourteen current
models, routing policies, organization members, budgets, keys, and ~26k usage
rows, with a side-by-side Playground comparison streamed from a mock upstream.

## Regenerate

```bash
bash scripts/demo_gif/record.sh
```

That seeds a throwaway SQLite gateway, boots it (and the mock provider on :8099)
serving the built dashboard bundle, drives the tour with Playwright while
recording, and encodes the result to `assets/otari-demo.gif`. Ports 8000 and 8099
must be free.

## Prereqs

- `ffmpeg` on `PATH` (webm to gif).
- `gifsicle` on `PATH` (lossy GIF optimisation).
- Web deps installed: `(cd web && pnpm install --frozen-lockfile)` (Playwright resolves from
  `web/node_modules`).
- A built dashboard bundle in `src/gateway/static/dashboard` (gitignored, so a
  fresh clone has none): `make dashboard`. Rebuild after any change under
  `web/src` or to `docs/dashboard.md` (bundled into the app), otherwise the
  recording shows the previous build.
- Network access: the Models page reads model metadata (vendor, context length,
  modalities) from models.dev at runtime, and shows bare ids without it.
- On arm64 Linux, `record.sh` sets the Playwright platform override automatically; on x86_64 it leaves Playwright's native detection and host checks in place.

## Files

| File | Purpose |
| --- | --- |
| `otari.yml` | Gateway config (SQLite, fixed master key `otari-demo-key`): providers, their models, routing policies. |
| `seed.py` | Deterministic seed: organization members, budgets, keys, pricing, usage (part of it routed). |
| `mock_provider.py` | OpenAI-compatible stand-in upstream that streams canned replies for the Playground stop. |
| `tour.mjs` | Playwright script: signs in and walks the tour with a drawn cursor, recording video + per-stop screenshots. |
| `record.sh` | Orchestrates seed to boot to record to encode. |

## Tweaking

- **Data**: edit the tables at the top of `seed.py` (models, policies, budgets,
  members, keys) and keep `MODELS` and `POLICIES` in step with `otari.yml`. RNG
  is seeded (`4242`) so runs are reproducible. The newest rows (`RECENT`) are
  written out by hand, since they are the ones Overview and Activity open on.
- **Pace / route order / dwell**: edit the stops in `tour.mjs`, one block per
  page. Each waits for its page heading, so a renamed page fails the run rather
  than recording a broken tour.
- **Playground replies**: edit `REPLIES` in `mock_provider.py`; the tour waits
  for a phrase from the OpenAI one, so change both together.
- **Size / fps / quality**: edit `FPS`, `WIDTH`, and the palette flags in
  `record.sh`.

Intermediate output (recording, per-page screenshots, palette, server log)
lands in `scripts/demo_gif/artifacts/` (gitignored) for inspection.
