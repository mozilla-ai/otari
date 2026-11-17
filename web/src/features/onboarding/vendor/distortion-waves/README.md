## distortion-waves (vendored)

The engine behind the first-run sheet's landing wave (`../../setupTransition.ts`),
copied from [Jakubantalik/Libraries.dev](https://github.com/Jakubantalik/Libraries.dev/tree/371b324a253b8ee71b9ce1bc8a1ecc5a59e1a16a/packages/distortion-waves/src)
at commit `371b324a253b8ee71b9ce1bc8a1ecc5a59e1a16a`, MIT (see `LICENSE`).

Vendored because the React package is not on npm yet. Only the four modules
the controller needs are here; the React wrapper and the barrel are not.

The files are upstream's but for one patch, marked "Patched in otari" in
`capture.ts`. Tailwind's preflight gives every box `border: 0 solid`, and
upstream's picture of the DOM kept the solid style but dropped the zero width,
so every box in the bent picture grew a 3px gray border. Biome and the
semantic-token check skip this directory so the copy stays comparable with
upstream. Do not edit it otherwise.

When `distortion-waves` is published with that fixed, add it to
`package.json`, point the three imports in `setupTransition.ts` (two dynamic,
one type-only) at `"distortion-waves"`, and delete this directory.
