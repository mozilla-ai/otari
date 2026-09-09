# Component catalog

The design system as running code: every component in `src/design-system/` with
its variants, its states and both themes, one story per axis.
[web/design/DESIGN.md](../design/DESIGN.md) is the specification; this is the
thing you look at.

## Run it

    pnpm --dir web run storybook         # dev server on :6006
    pnpm --dir web run storybook:build   # static build into storybook-static/

## Published

`.github/workflows/otari-design-system.yml` publishes it to GitHub Pages from
`main`. A pull request touching `web/` is gated on the catalog *building*, which
is where a story that stops compiling fails, and that takes about a minute. The
smoke sweep below runs on main and on demand rather than per PR: it takes
seconds locally and over twenty minutes on a hosted runner, which is six times
the slowest existing job. So run it locally before pushing, which is where it is
fast.

`STORYBOOK_BASE_PATH` is why the published build resolves its assets: Pages
serves a project site under `/<repo>/`, and the merged `vite.config.ts` carries
the app's own `base: "/"`. `main.ts` reads that variable and leaves the base
alone when it is unset, so a local build and the dev server need nothing.

## Smoke test

    pnpm --dir web exec node .storybook/smoke.mjs   # with a catalog served on :6006

Renders every story in headless Chromium, in both themes, and reports any that
error, render nothing, or log an uncaught exception. Playwright is already a dev
dependency here, so this needs nothing extra. Around ten seconds for the whole
catalog against a static build.

Two knobs, both with working defaults: `SMOKE_ORIGIN` points it at another port,
and `SMOKE_CONCURRENCY` sets how many pages drain the work list (6). The pool is
what makes it quick, and the reason it exists is worth keeping: as two sequential
loops over one page each, the same run took over 45 minutes on a CI runner and
printed nothing until it finished, so a slow run and a hung one looked identical.
It now prints a progress line every 50 renders.

**It catches what it claims to.** Verified by breaking a story on purpose: the
throw was reported against `design-system-indicators-kbd--default` in both
themes, by name and with its message. Worth repeating that positive control after
any change to `RENDERED`, because every failure mode here is silent by nature.

Three traps it was written around, all worth keeping:

- `#error-message` is always in Storybook's DOM, so it is no signal. The body's
  `sb-show-main` / `sb-show-errordisplay` class is.
- Several stories wrap their subject in a sizing `<div>`, and a modal portals
  out of `#storybook-root` entirely. So it waits for real content anywhere, not
  for `childElementCount`.
- **Some primitives render no text and no SVG at all.** A toggle's track, a
  divider's hairline, a meter's bar and a skeleton's block are correct and
  entirely graphical, so a text-or-SVG check reported nine of them as timeouts.
  `RENDERED` therefore also accepts something that *paints*: a descendant with a
  non-zero box and either a fill or a border. Deliberately not a child count,
  which the sizing `<div>` above would satisfy while the component inside is
  still null.

## Layout

- `main.ts`: story glob, the `viteFinal` that strips the TanStack Router plugins
  so a Storybook run can never rewrite `src/routeTree.gen.ts`, and the
  `STORYBOOK_BASE_PATH` seam.
- `preview.tsx`: imports `globals.css` (the whole design system) and composes
  the decorators. Note the order: innermost first.
- `theme.tsx`: the light/dark toolbar, writing the same three properties on
  `<html>` that `useTheme.tsx` and `index.html` do.
- `appContext.tsx`: `DeploymentProvider` and `SelectedWorkspaceProvider`, for
  the feature components that read them. Override the bootstrap per story with
  `parameters.deployment`.
- `apiMock.tsx`: a `fetch` stub driven by `parameters.api`. Failures use a
  `{ $status, $body }` envelope; see the comment there for why it is
  `$`-prefixed rather than sniffed.

Config, not stories, lives here. Stories sit beside the component they document,
the way `Foo.test.tsx` sits beside `Foo.tsx`.

## Titles

Two roots, and which one a story belongs to is not a judgement call: it is the
layer its subject lives in.

- **`Design system/<Topic>/<Name>`** for `src/design-system/`, where `<Topic>`
  is the directory, which is itself named for the file in `web/design/` that
  specifies it.
- **`Dashboard/<Feature>/<Name>`** for `src/features/` and the two directories
  left in `src/shared/components/`. These are application components: they read
  the deployment, the transport, or a domain type, and they are not part of the
  library.

A story's group therefore says whether its subject could leave in a package.
