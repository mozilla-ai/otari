# Component catalog

## Run it

    pnpm --dir web run storybook         # dev server on :6006
    pnpm --dir web run storybook:build   # static build into storybook-static/

## Smoke test

    pnpm --dir web exec node .storybook/smoke.mjs   # with the dev server running

Renders every story in headless Chromium, in both themes, and reports any that
error, render empty, or log an uncaught exception. Playwright is already a dev
dependency here, so this needs nothing extra.

Two traps it was written around, both worth keeping:

- `#error-message` is always in Storybook's DOM, so it is no signal. The body's
  `sb-show-main` / `sb-show-errordisplay` class is.
- Several stories wrap their subject in a sizing `<div>`, and a modal portals out
  of `#storybook-root` entirely. So it waits for real text or SVG anywhere,
  not for `childElementCount`.

## Layout

- `main.ts`: story glob, and the `viteFinal` that strips the TanStack Router
  plugins so a Storybook run can never rewrite `src/routeTree.gen.ts`.
- `preview.tsx`: imports `globals.css` (the whole design system) and composes
  the decorators. Note the order: innermost first.
- `theme.tsx`: the light/dark toolbar, writing the same three properties on
  `<html>` that `useTheme.tsx` and `index.html` do.
- `appContext.tsx`: `DeploymentProvider` and `SelectedWorkspaceProvider`, for
  the feature components that read them. Override the bootstrap per story with
  `parameters.deployment`.
- `apiMock.tsx`: a `fetch` stub driven by `parameters.api`. Failures use a
  `{ $status, $body }` envelope; see the comment there for why it is `$`-prefixed
  rather than sniffed.

Stories live beside their component under `src/`, so `pnpm run lint`,
`pnpm run typecheck` and `src/styles/foundation.test.ts` all cover them, which
is why a story styles from semantic tokens like anything else in the tree.
