# Otari Router Demo

A small React app (Vite + HeroUI v3 + Tailwind v4) that demonstrates the Otari
kNN router. It is fully offline: everything runs against a bundled set of
prompts, model answers, and embeddings (`src/demo_prompts.json`). There is no
gateway connection and no OpenAI call at runtime.

It has two tabs:

- **How it learns** — a read-only walkthrough: for a sample prompt it shows each
  model's answer and the gpt-5.4 judge's quality score, then how those become
  routing-memory records and which cheaper model the router would aim to serve.
- **See it route** — runs the shipped kNN scoring formula as leave-one-out over
  the bundled prompts. Pick a prompt, route it, and see its nearest neighbors,
  the per-model scores, and which model the router picks. A cost dial (alpha)
  shifts the cheap-vs-premium boundary.

The bundled answers come from a four-tier ladder with a deliberately wide
capability gap: **gpt-5.4** and **gpt-4o** (premium), **gpt-4o-mini** (cheap,
capable), and **gpt-3.5-turbo** (cheapest, weakest). Prices are illustrative and
increase with capability so no tier is strictly dominated.

## Run

```bash
cd demo/router
npm install
npm run dev        # opens on http://localhost:5180
```

## Build

```bash
npm run build      # type-checks and emits a static bundle in dist/
npm run preview    # serves the built bundle locally
```

The `dist/` bundle is plain static files; drop it on any static host (GitHub
Pages, S3, nginx, etc.). No server or API key is involved.

## Notes

- **Blind ranking** (Settings, on by default) hides which model wrote each answer
  until you submit, so you judge the text, not the brand.
- Settings persist in `localStorage`; nothing is sent anywhere.
- The bundled `src/demo_prompts.json` is organized into semantic clusters: a
  general-knowledge region where both models succeed (so the router goes cheap)
  and harder code and math/puzzle regions where gpt-3.5-turbo falls short (so the
  router reaches for gpt-5.4). The kNN routes by neighborhood, so keeping coherent
  clusters is what makes routing visible. Each answer is scored 0 to 1 by a
  gpt-5.4 judge. Regenerate it with `demo/router/generate_demo_dataset.py` (needs
  `OPENAI_API_KEY`).
