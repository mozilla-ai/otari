# Design tokens: `web/`

**The tokens are the design system.** HeroUI and Tailwind are consumers of it, not sources of
truth for it: HeroUI's internal variables are aliased to our `--color-*` tokens, so a bare
`<Card>` wears our palette, and every utility a component writes has to resolve back through
that mapping. That is the whole point. Retheming the dashboard, or dropping a different
component library under it, is then an edit to one file rather than a sweep through the tree.
A utility that paints a color without passing through a token, whether it is a Tailwind
palette class or one HeroUI happens to ship, is outside the system and will not follow the
next change to it.

The visual system is the one rehomed from `otari-ai/frontend`, which the architecture charter
settles as canonical for the converged dashboard. All of it lives in
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

That is Tailwind v4's own model rather than a local invention: a variable in `@theme` generates
the utilities for its namespace (`--color-*` the color utilities, `--radius-*` the `rounded-*`
family, `--font-*` the font ones), which is why a token is consumed as a utility and not by
name. One mechanic is worth knowing before you add a token that is an alias of another. Tailwind
asks for `@theme inline` when a theme variable references a second variable, because a
non-`inline` entry is substituted where it is declared, at `:root`, rather than at the element
using it. This file's `@theme` block is not `inline`, and the reason it still follows a theme
switch is the convention that every theme block declares the complete set: `--color-attention`
and friends are self-referential registrations whose real values live in the light and dark
blocks, and an alias like `--color-background-alt` is re-declared in each block rather than only
in `@theme`. So when you add one, put it in both theme blocks (which `foundation.test.ts`
enforces) or make the registration `inline`; a lone `@theme` alias resolves once, at the root,
and stops tracking the theme.

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

**Use ours.** The test is not whether a utility's name came from HeroUI, it is whether the
value behind it does. `bg-accent`, `text-foreground`, `text-muted`, `border-border` and
`bg-default` are HeroUI-named and correct, because the theme block above aliases each of
those variables to a `--color-*` token of ours. The families listed here are the opposite
case: nothing in this repo declares them, so they resolve to whatever `@heroui/styles` ships
and will not follow a token change. The `-subtle` and `-alt` names differ from HeroUI's
because otari-ai chose them before this rehome, and renaming here would fork the file the M5
merge depends on.

Two traps follow from the same overlap. A class that *looks* like it belongs to one family may
belong to neither and silently emit nothing, which is what happened to `bg-danger-subtle` and
`text-info` before they were registered, and to `bg-content1`, a HeroUI **v2** name that v3
neither declares nor reads. When adding a utility, add it to `DOCUMENTED_UTILITIES` in
`src/styles/foundation.test.ts`, and confirm it compiles by building and grepping the emitted
stylesheet. A class that does not exist raises nothing at any stage.

### Porting a component from otari-ai

Both repos alias HeroUI's internals to the same `--color-*` tokens, so the palettes agree.
The **utility vocabulary does not**, and that is the thing to translate when a component moves
between the trees:

| otari-ai writes | Write here | Token behind it |
| --- | --- | --- |
| `bg-content1` | `bg-surface` | `--color-surface` |
| `bg-content2` | `bg-surface-alt` | `--color-surface-muted` |
| `bg-content3` | `bg-surface-subtle` | `--color-surface-subtle` |
| `bg-background-alt` | `bg-background-alt` | `--color-background-muted` |
| `bg-accent` | `bg-accent` | `--color-primary` |
| `text-foreground` | `text-foreground` | `--color-text` |
| `text-muted` | `text-muted` | `--color-text-muted` |

`bg-content1/2/3` are the only real translation, and they are worth knowing about for a second
reason: **they are dead classes in both repos.** `content1` is a HeroUI v2 name, v3's
`@heroui/styles` neither declares nor reads it, and Tailwind v4 generates a `bg-*` utility
only from a `--color-*` theme variable, which neither tree declares for it. Grepping
otari-ai's own built stylesheet for `.bg-content1` finds nothing while `.bg-surface` and
`.bg-accent` are both there, so the seventeen files using it over there are rendering with no
background at all rather than with `--color-surface`. Copying one of those class names here
would carry the bug across, silently, because a class that does not exist fails at no stage of
the build.

### The alias surface is not only color

The mapping above is color because color is what has needed it so far, not because that is as
far as it goes. HeroUI's own theming guide documents a wider surface, and `@heroui/styles`'s
README lists it under "CSS Variables": alongside the colors sit `--radius` (with `--radius-xs`
through `--radius-4xl` calculated from it in `dist/themes/shared/theme.css`, so `--radius-2xl`
is `calc(var(--radius) * 2)`), `--field-radius`, `--spacing`, `--border-width`,
`--field-border-width`, `--ring-offset-width`, `--disabled-opacity`, `--cursor-interactive` /
`--cursor-disabled` and the `--scrollbar-*` family, all declared in
`dist/themes/default/variables.css`. This repo aliases none of them, so a component's corner
radius, its disabled dimming and its pointer all come from HeroUI's defaults rather than from
anything we name.

**Only the color half belongs in both theme blocks.** A color token is declared twice because
its whole job is to hold a light value and a dark one, which is what lets a component adapt
without a `dark:` at the call site, and HeroUI says the same thing about its own ("define in
both light and dark themes"). A radius, a cursor, or a disabled opacity does not change between
themes, so it is one declaration, not a pair kept in sync for no reason. What the two halves
share is the rule that matters: the value lives in one named place, and a component reads it.
Never a hex, never a numbered Tailwind palette class, never `bg-white`, because those three do
not adapt and nothing downstream can make them.

That is a gap rather than a decision, and it decides how much work a visual fix is. A value
computed from a variable is one alias away from being ours; the same value chased through the
rules that read it is a selector to keep in sync with somebody else's internals, forever. A
table whose body corners are drawn at `min(32px, var(--radius-2xl))`, which is 16px, inside a
12px container is the case that made this concrete: flattening them through
`.table__body tr:first-child td:first-child` works, and setting the variable on the table root
does the same job to every rule that reads it, including the ones nobody has hit yet. So when
something looks wrong, search `node_modules/@heroui/styles/dist/` for the property before
writing a rule against it. If a variable is behind it, set it where its scope actually is: with
the theme when the decision is system-wide (adding a role to the families above if it is one of
ours), or on the component's own root when it is local. Scoping a variable to a subtree is our
extension rather than something HeroUI's guide demonstrates, and it inherits, so check what else
inside that subtree reads it, and scope **the variable the rule reads**: a derived custom
property is substituted where it is declared, so `--radius-2xl` computed at `:root` keeps the
root's value inside a subtree that redefines `--radius`.
[components.md](./components.md) has the full order to work
through.

## Type scale

Seven roles, each an `@utility` in the same file. Pick the one whose **meaning** matches the
text, not its size:

`text-display` (one per route) · `text-heading` (section) · `text-title` (card/dialog) ·
`text-body` (default, matches `<body>`) · `text-emphasis` (rare) · `text-caption` (metadata) ·
`text-overline` (small uppercase group label).

The scale is the whole set of sizes. `text-[11px]` is `text-overline` spelled in a way the
system cannot see, `text-2xl font-bold` outside `text-display` means the hierarchy is the bug,
and a one-off combination (`text-base font-semibold uppercase tracking-tight`) that repeats is
a role that should be added here instead. One `text-display` per route. Heading *level* is a
separate decision from heading *style*: `<h1>`/`<h2>`/`<h3>` follow the document outline, so a
card title nested two sections deep may be an `<h3>` or an `<h4>` and still wear
`text-title`. Pick the level that keeps the outline unbroken, then the role that matches the
meaning.

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
HubSpot patch dress a public site and a hosted signup that do not exist here. The
first-request setup guide *is* ported (`features/onboarding/SetupGuideCard.tsx`), and it
deliberately dresses itself in the status and surface families rather than bringing that
bespoke `activation-*` set with it: a panel on the Overview page is a card with a status
row, so the roles the foundation already names are the right ones, and a family with one
consumer is a second palette to keep in step for no reader benefit.

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

The namespace existing is not the same as the approach being recommended. HeroUI supports a
rule against its own classes, so the reason this one is last of the four ways to change how
something looks, after a variable, a wrapper or utility, and the component's own props, is not
permission: a rule fixes one selector where a variable fixes every rule that reads it, it is
invisible from the call site, and because these rules are unlayered they outrank a Tailwind
class at the call site too, so they take away the ability to restyle from the place that would
most naturally do it. [components.md](./components.md) has the order and the reasoning. When a
rule really is the only way, name in its comment which of the three does not reach the value.
