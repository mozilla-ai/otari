# Bundled brand fonts

Self-hosted `woff2` faces for the three type roles the design foundation declares
(`--font-sans`, `--font-display`, `--font-mono` in `web/src/styles/globals.css`).
Self-hosted rather than loaded from a CDN so the dashboard renders its own
typography with no third-party request, which also means an air-gapped gateway
looks the same as a connected one.

Every family here is licensed under the SIL Open Font License 1.1. The OFL
requires the copyright notice and the license text to travel with the font, so
each family's license ships in this directory next to the faces it covers. Vite
copies `public/` verbatim into the bundle, so the license files are served from
the running dashboard too, not only present in the source tree.

| Family | Faces | License | Copyright |
| --- | --- | --- | --- |
| Mozilla Text | `MozillaText-Variable.woff2`, `MozillaTextItalic-Variable.woff2` | [OFL 1.1](MozillaText-OFL.txt) | Copyright 2024, The Mozilla Foundation |
| Zilla Slab | `ZillaSlab-Medium.woff2`, `ZillaSlab-SemiBold.woff2`, `ZillaSlab-Bold.woff2` | [OFL 1.1](ZillaSlab-OFL.txt) | Copyright 2017, The Mozilla Foundation |
| Fira Code | `FiraCode-VariableFont_wght.woff2` | [OFL 1.1](FiraCode-OFL.txt) | Copyright (c) 2014, The Fira Code Project Authors |

The license and copyright of each row are the font binary's own: the `name`
table of every file here carries the copyright string in the table above and a
license description naming OFL 1.1. `MozillaText-OFL.txt` is the canonical OFL
1.1 text under the copyright line and license URL that font declares, because
Mozilla Text has no public upstream repository to copy a `LICENSE` from; the
other two are their upstream projects' own license files verbatim.

Adding a face means adding its license here in the same commit. `fonts.test.ts`
fails when a `.woff2` in this directory has no license file covering it, so the
pair cannot come apart later.
