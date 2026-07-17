# Design tokens: `web/`

The dashboard's palette is a small set of CSS custom properties defined once, at the top of
`web/src/styles/globals.css`:

```css
:root {
  --otari-brand: #4e8295;
  --otari-brand-dark: #3c6678;
  --otari-brand-tint: #eaf1f3;
  --otari-ink: #14242c;      /* primary text */
  --otari-muted: #5b6b73;    /* secondary text */
  --otari-line: #dbe5e8;     /* borders/dividers */
  --otari-surface: #ffffff;  /* cards/panels */
  --otari-bg: #f6f9fa;       /* page background */
}
```

Consume them through Tailwind arbitrary values:

```tsx
<span className="text-[var(--otari-muted)]">…</span>
<div className="border border-[var(--otari-line)] bg-[var(--otari-surface)]">…</div>
<h1 className="text-[var(--otari-ink)]">…</h1>
```

## Rules

- **Add a token, don't scatter a hex.** If you need a new brand or surface color, add a
  `--otari-*` variable in `globals.css` and reference it. A one-off hex in a component is a
  review smell: the next person can't retheme the app from one place.
- **Name by role, not appearance.** `--otari-muted`, `--otari-line`, `--otari-surface` describe
  what the color is for. Don't add `--otari-teal-500` or `--otari-gray`; if you're tempted, the
  role is missing, not the shade.
- **Status surfaces are the documented exception.** Alert/banner components
  (`ErrorBanner`, `InfoBanner` in `components/ui.tsx`) use raw Tailwind palette classes
  (`red`, `amber`) for their semantic meaning. Match that for new status elements; don't
  convert them to `--otari-*` and don't invent brand tokens for them.
- **No general-purpose palette classes for chrome.** `bg-white`, `text-gray-900`, `bg-zinc-800`
  for ordinary surfaces/text should be a `--otari-*` variable instead.

## HeroUI theming

HeroUI v3 ships its own styles (`@import "@heroui/styles"` in `globals.css`) and its `dark`
variant is wired via `@custom-variant dark (&:is(.dark *))`. Let HeroUI components carry their
own theming through `variant`/`size` props rather than overriding their internals with
`className`. If a HeroUI surface needs to match an `--otari-*` color, prefer a prop or a small
wrapper over reaching into the component's classes.
