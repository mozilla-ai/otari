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
| Brand | `--color-primary`, `--color-primary-hover`, `--color-primary-active`, `--color-primary-subtle`, `--color-primary-foreground`, `--color-primary-subtle-foreground` |
| Link & focus | `--color-link`, `--color-link-hover`, `--color-focus` |
| Status | `--color-success`, `--color-warning`, `--color-danger`, `--color-info`, each with a `-subtle` fill and a `-foreground` |
| Attention | `--color-attention` (+ `-subtle`, `-foreground`, `-border`) |
| Code | `--color-code-surface`, `--color-code-border`, `--color-code-control`, `--color-code-foreground` |
| Elevation | `--shadow-sm/md/lg`, `--shadow-modal` |
| Fields | `--field-background`, `--field-border`, `--field-border-width` |
| Data viz | `--color-chart-cat-1` … `-8` + `-other`, `--color-chart-ramp-1` … `-4` |

## How they reach the screen

Don't consume the tokens by name. Each theme block aliases HeroUI's own variables to them
(`--background` → `--color-background`, `--surface` → `--color-surface`, `--default` →
`--color-background-muted`, `--accent` → `--color-primary`, and so on), which is what makes a
bare HeroUI component wear the palette with no styling at the call site. Consume the Tailwind
utilities that resolve through that mapping:

```tsx
<span className="text-muted">…</span>
<div className="border border-border bg-surface">…</div>
<p className="text-danger">…</p>
<Card>…</Card>   {/* already on --color-surface, with a 1px --color-border outline */}
```

Tokens HeroUI has no counterpart for are registered in the file's `@theme` block, which is
what generates their utilities: `bg-background-alt`, `bg-surface-alt`,
`bg-surface-subtle`, `text-link`, `text-link-hover`, the `attention` and `info` families, the
status `-subtle` fills, `bg-primary-subtle`, `text-primary-subtle-foreground`, the `code`
family, `shadow-elevation-sm/md/lg`, `shadow-modal`, and `font-sans` / `font-mono` /
`font-display`.

Two of those name a pairing rather than a color, and are the ones easiest to get wrong.
**Text on a `-subtle` fill is not the fill's own color**: `text-primary-subtle-foreground` is
what goes on `bg-primary-subtle` (the brand teal on the brand tint is 3.8:1, under AA), while
`text-danger` / `text-warning` / `text-success` / `text-attention` *are* the right ink on
their own subtle fills, because those four are tuned to clear 4.5:1 there. And the second is
that a hover step has to move away from what it sits on: the shell chrome is painted with
`bg-surface`, so a nav item hovers to `bg-surface-alt`. `--color-background-muted` is the
token whose comment claims the chrome, and the shell deliberately does not use it, because
`bg-primary-subtle` and that silver are a shade apart and an active nav item disappears
against it.

### HeroUI ships a near-synonym for several of these

`@heroui/styles` has its own `@theme`, so `bg-danger-soft`, `bg-accent-soft`,
`bg-default-soft`, `bg-background-secondary`, `bg-background-tertiary` and
`border-border-secondary` are all real classes that nothing in this repo declares. They
overlap in meaning with our `-subtle` and `-alt` names without being the same thing, and the
compiler cannot tell you which family a class came from: both spellings work, and only one
tracks our tokens.

**Use ours.** A HeroUI-named utility bypasses the palette and will not follow a token change.
The names differ because otari-ai chose `-subtle` and `-alt` before this rehome, and renaming
here would fork the file the M5 merge depends on.

Two traps follow from the same overlap. A class that *looks* like it belongs to one family may
belong to neither and silently emit nothing, which is what happened to `bg-danger-subtle` and
`text-info` before they were registered, and to `bg-content1`, a HeroUI **v2** name that v3
neither declares nor reads. When adding a utility, add it to `DOCUMENTED_UTILITIES` in
`src/styles/foundation.test.ts`, and confirm it compiles by building and grepping the emitted
stylesheet. A class that does not exist raises nothing at any stage.

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
## Chart colors

Series fills are their own family because a mark is read against a plot, not against text:
WCAG says nothing about whether two adjacent bars are separable, and CVD separation says
nothing about legibility. Both sets are validated **per theme** with the data-viz palette
checker, so a value there is a checked result rather than a shade someone liked.

- `--color-chart-cat-1` ... `-8` are the categorical slots for grouped series. Assign them in
  that order and **never cycle**: the margins are computed on adjacent pairs in exactly this
  order. A ninth group folds into `--color-chart-cat-other`.
- `--color-chart-ramp-1` ... `-4` are one hue at four lightnesses, **strongest first**, for a
  part-to-whole bar whose segments are ordered rather than categorical (the billed token
  composition on Activity and Usage). Light runs dark to pale, dark runs pale to dark.

Consume them as `var(--color-chart-...)` in the `SeriesDef.color` or SVG `fill` a chart takes,
not as a Tailwind class; they are deliberately not registered in `@theme`, because no
component should be painting a background with a series color. Changing one means re-running
the checker for both themes, not adjusting it by eye.

## The retired `--otari-*` palette

The dashboard used to have its own pre-rehome palette (`--otari-brand`, `--otari-ink`,
`--otari-muted`, `--otari-line`, the categorical chart slots) behind a `MIGRATION BRIDGE`
marker in `globals.css`, with the hand-rolled primitives that consumed it. It is gone: every
page is on the foundation, and `foundation.test.ts` fails if an `--otari-*` variable is
declared again. If you find yourself wanting one, the role is missing from the foundation, so
add the role.

The `otari-` prefix that survives on a handful of **class** names (`.otari-table`,
`.otari-markdown`, `.otari-detail-row`, `.otari-bulk-bar`) is unrelated: it is the app's
namespace for a hook that has to reach inside a HeroUI component's DOM, and those rules
consume `--color-*` like everything else.
