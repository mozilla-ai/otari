# TODOS

Deferred work with a recorded reason. Items land here from plan reviews so a
deferral is a decision with a rationale, not a dropped thread.

## From issue #525 (Usage share card)

- **Key-scoped share for non-admin developers.** `GET /v1/usage/summary` is
  `Depends(verify_master_key)` (`src/gateway/api/routes/usage.py:1320`) and the
  dashboard authenticates only as the master key, so today exactly one person per
  deployment can generate a share card. The issue's audience ("people like to share
  their token usage") is individual developers holding an API key. The
  `verify_api_key_or_master_key` pattern already exists (`usage.py:482`) and would
  turn one poster per gateway into every developer with a key. Backend auth work,
  deliberately out of the v1 blast radius. **Priority: P1, this is the reach ceiling.**

- **Hosted share page with `og:image`.** A PNG is not clickable, so the growth loop
  terminates at whatever the poster types. A real URL is trackable and renders
  identically for every viewer. The tradeoff against it: a public unauthenticated
  route on a gateway that manages API keys, and self-hosted gateways are often not
  internet-reachable. Would live on otari.ai rather than in the gateway. **P2.**

- **Success metric for the share feature.** The dashboard carries no analytics by
  design, so this is manual: count share-card posts in the wild at 30 and 90 days
  after ship. **If fewer than 10 distinct people have posted one by day 90, delete
  the feature** rather than carry `ShareCard` + `ShareDialog` + `shareImage` + 6 test
  files and a rasterization path through every future HeroUI and Tailwind major.
  **Delete-by review date: 2026-11-11.**

- **Usage page URL deep-link share.** `UsagePage` holds filters in `useState`, so a
  Usage view cannot be shared as a link, unlike Activity which already uses
  `urlState.ts`. Cheaper than the image card and complementary to it. **P2.**

- **OverviewPage share.** Natural follow-up once the card exists. **P3.**

- **Retire the client-side open-weights fallback.** If the `provider_model` summary
  dimension ships with v1, the suffix-matching path and the "unclassified" bucket
  exist only for older gateways. Remove once the supported floor moves past that
  release. **P3.**

- **X prefers square, LinkedIn prefers 1.91:1.** Recorded so the ratio default
  (square 1080x1080) is a decision rather than an accident. **P3, informational.**
