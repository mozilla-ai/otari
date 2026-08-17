# Design tokens: `web/`

The dashboard's visual system is the one rehomed from `otari-ai/frontend`, which the
architecture charter settles as canonical for the converged dashboard. All of it lives in
`web/src/styles/globals.css`: the self-hosted brand faces, the tokens, the type scale, the
base reset, and a short list of HeroUI v3 patches.

## The token families

Every token is named for its **role**, declared twice (a light block and a dark one), and
never for its appearance:

| Family | Tokens |
| --- | --- |
| Surfaces | `--color-background`, `--color-background-muted`, `--color-background-subtle`, `--color-surface`, `--color-surface-muted`, `--color-surface-subtle` |
| Text | `--color-text`, `--color-text-muted`, `--color-text-subtle`, `--color-text-on-primary`, `--color-text-on-inverse` |
| Borders | `--color-border`, `--color-border-strong`, `--color-border-subtle` |
| Brand | `--color-primary`, `--color-primary-hover`, `--color-primary-active`, `--color-primary-subtle`, `--color-primary-foreground` |
| Link & focus | `--color-link`, `--color-link-hover`, `--color-focus` |
| Status | `--color-success`, `--color-warning`, `--color-danger`, `--color-info`, each with a `-subtle` fill and a `-foreground` |
| Attention | `--color-attention` (+ `-subtle`, `-foreground`, `-border`) |
| Code | `--color-code-surface`, `--color-code-border`, `--color-code-control`, `--color-code-foreground` |
| Elevation | `--shadow-sm/md/lg`, `--shadow-modal` |
| Fields | `--field-background`, `--field-border`, `--field-border-width` |

## How they reach the screen

Don't consume the tokens by name. Each theme block aliases HeroUI's own variables to them
(`--background` → `--color-background`, `--content1` → `--color-surface`, `--default` →
`--color-background-muted`, `--accent` → `--color-primary`, and so on), which is what makes a
bare HeroUI component wear the palette with no styling at the call site. Consume the Tailwind
utilities that resolve through that mapping:

```tsx
<span className="text-muted">…</span>
<div className="border border-border bg-content1">…</div>
<p className="text-danger">…</p>
<Card>…</Card>   {/* already on --color-surface, with a 1px --color-border outline */}
```

Tokens HeroUI has no counterpart for are registered in the file's `@theme` block, which is
what generates their utilities: `bg-background-alt`, `bg-surface-alt`, `text-link`,
`text-link-hover`, the `attention` family, the `code` family, `shadow-elevation-sm/md/lg`,
`shadow-modal`, and `font-sans` / `font-mono` / `font-display`.

## Type scale

Seven roles, each an `@utility` in the same file. Pick the one whose **meaning** matches the
text, not its size:

`text-display` (one per route) · `text-heading` (section) · `text-title` (card/dialog) ·
`text-body` (default, matches `<body>`) · `text-emphasis` (rare) · `text-caption` (metadata) ·
`text-overline` (small uppercase group label).

Headings and the display/heading roles are set in Zilla Slab; body and UI text in Mozilla
Text; keys, IDs, and code in Fira Code. The faces are self-hosted in `web/public/fonts/` under
SIL OFL 1.1, with each family's license shipped beside it (see the README there), so the
dashboard's typography needs no third-party request and an air-gapped gateway looks like a
connected one. Adding a family means adding its license in the same commit;
`src/styles/fonts.test.ts` fails on a face with no license covering it, a shipped face never
declared, or a declared face never shipped. Self-hosting also depends on the gateway actually
serving `/fonts`, which is three pieces of plumbing outside `web/`; the Serving bullet in
[web/AGENTS.md](../../../web/AGENTS.md) says which, and `tests/unit/test_gateway_root_page.py`
covers them.

## What stayed in otari-ai

Its marketing type roles, pre-login gradient, activation-modal and border-beam tokens, and
HubSpot patch dress a public site and a hosted signup that do not exist here.

Its `shared/components/ui/adapters/` layer stayed too, and that one is worth knowing about,
because it is the obvious thing to reach for and it should not be reached for. It is a HeroUI
**v2 to v3 compat shim** whose own header marks it for deletion call site by call site over
there. This dashboard was written against v3 and has no v2 call sites, so importing it would
mean adopting `react-icons` and a react-hook-form ref bridge to gain a layer already being
retired, and handing new components a v2-flavored API. Most of that repo's primitives
(`EmptyState`, `ErrorBoundary`, `HelpIcon`, `CodeBlock`, `ConfirmModal`, `FormModal`,
`ResponsiveTabs`, `YesNoButtonGroup`) bind to the shim or to `react-icons`, so they wait on the
same decision. `SettingsSection` and `RowActions`, which need neither, came across unchanged.

## Rules

- **Add a token, don't scatter a hex.** A one-off hex in a component is a second source of
  truth for a color the tokens already name; the next person can't retheme the app from one
  place. `src/styles/foundation.test.ts` rejects one outright under `shared/components/ui/`.
- **Add it to both theme blocks.** Each block owns the complete set it needs rather than
  inheriting from its sibling, so a token declared in one only falls back to the other
  theme's value, which shows up as a contrast bug pages away from the edit. The same test
  fails when the two sets diverge.
- **Name by role, not appearance.** `--color-text-muted`, `--color-border`, `--color-surface`
  say what the color is *for*. Never `--color-teal-500` or `--copper-300`; if you're tempted,
  the role is missing, not the shade.
- **No numbered Tailwind palette classes.** `bg-emerald-50`, `text-gray-900`, `border-red-200`
  bypass the whole system and are invisible to the dark theme. A status color is a token.
- **Keep it recognizable as `otari-ai/frontend/src/index.css`.** The control-plane pages land
  here at M5 and have to land on this palette without a reconciliation pass, so a divergence
  in a token's name, value, or reasoning comment costs more than it looks like it does.

## The `--otari-*` bridge

Everything below the `MIGRATION BRIDGE` marker in `globals.css` is the dashboard's
pre-rehome palette (`--otari-brand`, `--otari-ink`, `--otari-muted`, `--otari-line`, the
categorical chart slots) plus the component rules that consume it. It is kept, at its
original values, only so the existing call sites keep rendering while their pages are rebuilt
on the foundation. Same for the files in `web/src/shared/components/` carrying the matching
header comment.

Nothing new may consume an `--otari-*` variable and nothing new may be added to the block;
`foundation.test.ts` fails if one is declared above the marker. Don't "fix" the values toward
the foundation either: a half-converted palette is worse than two clearly separated ones, and
converting a page is what retires its tokens. When the last call site goes, so does the whole
section.
