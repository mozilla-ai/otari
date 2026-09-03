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

Use pnpm as pinned in `package.json`. The Docker build stage is Node 26 and CI
tracks the `lts/*` alias, so `@types/node` follows the image at 26. Keep shared
React Aria dependencies on the version HeroUI resolves, and approve required
install scripts in `pnpm-workspace.yaml`.

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

Regenerate the icons with `pnpm --dir web run icons:generate`
(`web/scripts/generate-pwa-icons.mjs`). It rasterizes the mark through
Playwright's own Chromium rather than adding an image-processing dependency,
reads the target sizes from the manifest, and centers the mark by its 273x250
aspect so it is never stretched.

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

**A failure's accessibility snapshot describes the state at capture, not at the
action.** It is written when the assertion finally times out, which can be
seconds after the click that actually failed, so it shows the page as it settled
rather than as it was pressed. Reading correctly-loaded content in one and
concluding a race is refuted is a mistake this repo has already made: the
snapshot has no bearing on what was on screen when the action ran. Use the trace
instead, where each action carries its own duration and error, and a click that
completed in 68ms with no error tells you the press was delivered and the app
dropped it.

Assertions that press a row therefore wait on the data before pressing, not just
on the element. A press delivered while a filtered query is in flight is
discarded by the re-render while Playwright's actionability check passes
happily, because the row it is about to press is present and visible right up to
the moment it is replaced.

Screenshot tests cover three viewports and both themes. They are
workflow-dispatch only, their baselines are gitignored, and CI artifacts are the
review output until the suite becomes a pull-request gate.

## Measuring the running dashboard

Three ways a probe reports a real number that answers the wrong question. Each
has produced a wrong finding here more than once, and none of them errors.

**Check which bundle the tab is on first.** A gateway serves the last bundle
built into `src/gateway/static/dashboard/`, and a browser tab holds the one it
loaded, so a measurement can describe a build from an hour ago while the source
on disk says otherwise. Vite content-hashes every asset filename, which makes
the check cheap: build, note the emitted `assets/index-*.css`, and confirm the
tab is on that filename before trusting anything read from it.

**A computed style read in a background tab is frozen, not current.** While a
tab is not being rendered, `getComputedStyle` returns the start value of any
transition already running on the property and keeps returning it, so a scripted
state change reads back as the old color and a scripted press reads as no
transform at all. It looks like broken CSS rather than a sleeping instrument.
Suppress transitions before measuring anything that transitions
(`*{transition:none !important}` in an injected `<style>`, removed afterwards).
The same throttling stops `requestAnimationFrame` from firing, so a probe that
awaits a frame hangs instead of returning.

**A control's own box does not tell you whether it is visible.** The checkboxes
are drawn by a styled element beside a real `<input>` that react-aria hides, and
the hiding is on the input's wrapper. The input itself reports `opacity: 1`,
`position: static` and a 13x13 box, so a probe that measures inputs concludes
the product ships bare native checkboxes below the touch floor. It has concluded
that three times. Hit-test the element's own center with
`document.elementFromPoint`, or walk its `offsetParent` chain for a clipping
ancestor, before reporting geometry.

**The screenshot suite does not reach below the fold.** Its captures pass
`fullPage: true`, but the shell scrolls its content in an inner container rather
than scrolling the document, so a capture comes back at the viewport size
(1280x800 for the desktop projects) and anything further down the page is not in
it. The docs page is the clearest case: its one code fence sits below the fold
and appears in no screenshot project. Do not read a page's presence in that
suite as coverage of the whole page.

**Read the reporter, not the exit code, when a command is in a pipeline.** A
run piped into `tail`, `grep` or `head` exits with that stage's status, so a
suite with failures reports success and a green shell says nothing about the
tests. The same shape as the traps above from the other direction: those were
queries whose empty result read as a clean answer, this is a pipeline whose
last stage reported one. Check the failure count in the output, or keep the
command unpiped.

**A typecheck means nothing while any file has conflict markers.** One file
containing `<<<<<<<` suppresses every semantic diagnostic in the program, so
`tsc` reports only syntax errors and a clean-looking run mid-rebase proves
nothing. Reproduced on a two-file project: a real `TS2322` is reported alone and
disappears entirely once one unresolved file is added. Resolve the last conflict
before believing a typecheck, and treat lint and per-file test runs as the only
checks that hold before that.

**And absence from the built CSS proves nothing on its own.** Tailwind emits
only the utilities something in the tree asks for, so checking whether a
`@utility` or a token survived the build needs a consumer inside `src` that
references it. A consumer written outside the scanned tree produces an empty
result that looks exactly like a rule that failed to compile.
