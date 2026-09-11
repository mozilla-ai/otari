# Colors

Every color is a semantic token. Role names only: a token says what the color is
*for*, never what it looks like. `--copper-500` and `--warm-50` are banned names
as much as banned values.

Both themes are declared in `src/styles/globals.css`. A component never branches
on theme; it names a role and the theme block supplies the value.

## Surfaces: four rungs, two families, offset by one

| Write this class | Token | Light | Dark |
| --- | --- | --- | --- |
| `bg-background` | `--color-background` | `#f7f8f8` | `#08090a` |
| `bg-surface` | `--color-surface` | `#ffffff` | `#0f1011` |
| `bg-surface-alt` | `--color-surface-muted` | `#f0f1f2` | `#191a1b` |
| `bg-surface-subtle` | `--color-surface-subtle` | `#e8e9eb` | `#1f2023` |

**The class is not the token name.** `surface-muted` is the token; `bg-surface-alt` is
the class. `bg-surface-muted` is not a class and paints nothing. Read the left column.

Two pairs share a value with the `background-*` family. That is the stagger working,
not a duplicate to collapse.

What background do I use?

```text
Is it the page itself, the rail, or the top bar?
 ├── Yes -> bg-background   (the shell is flat; hairlines divide it, not levels)
 └── No
      ├── Is it a table body, a modal, or a field?
      │    └── Yes -> bg-surface
      ├── Is it a hovered row, or a ghost button's hover?
      │    └── Yes -> bg-surface-alt        (the utility for surface-muted)
      └── Is it a selected tab track, or a ghost button's press?
           └── Yes -> bg-surface-subtle
```

`bg-background-secondary` and `bg-background-tertiary` compile and paint nothing.
HeroUI registers the roles, this file declares neither variable. The working
spellings are `bg-surface-alt` and `bg-surface-subtle`.

## Text: three tiers

| Utility | Light | Dark | For |
| --- | --- | --- | --- |
| `text-foreground` | `#08090a` | `#f7f8f8` | Body and headings |
| `text-muted` | `#2c2e33` | `#c2c8d2` | Labels, captions, secondary lines |
| `text-subtle` | `#5a5f6b` | `#8a8f98` | Overlines, timestamps, helper text |

`text-subtle` has a second job: it is `--color-control-border`, the opaque edge of
every control. Retune it and every ghost button, switch and segmented track moves
with it. That coupling is deliberate and stated so nobody discovers it.

## Borders: alpha, never opaque

| Token | Value | For |
| --- | --- | --- |
| `border-border-subtle` | 3% | Rows inside one group |
| `border-border` | 6% | The hairline. Every band, every table rule |
| `border-border-strong` | 10% | A division that has to read as structural |
| `border-control-border` | opaque, 6.01:1 on the page | A control's own edge |

Alpha, because one line lands on four different rungs and must composite on all of
them. An opaque border tuned to the canvas is invisible on a card.

## The accent has five jobs and five values

`--color-primary` (`#0098a4`) cannot do all of them and clear AA. Pick by the job.

| Token | Job | Never |
| --- | --- | --- |
| `bg-accent` / `text-accent` | Data ink, active route, the mark | As body text: 2.87:1 |
| `--color-primary-button` | The filled CTA. 4.71 under white | For a chip or a meter |
| `text-link` | A link. The accent darkened until it reads | Without an underline |
| `--color-focus` | The focus ring, stepped down to clear 3:1 | Removed, ever |
| `bg-control-indicator` | A checkbox box, a toggle's on-knob | For a meter or a chart |

Ink **on** the accent is near-black (`--color-primary-foreground`), not white:
white measures 3.48:1 there and fails. That is the reverse of most accents and the
single most missed line in this file.

`bg-accent/…` is not how you get a tint. Use `--color-primary-subtle`, which is
translucent so it holds across all four rungs, and set text on it to
`--color-primary-subtle-foreground` (`#005b63`), because the accent itself is
under AA on its own tint.

## Status: four hues plus one warm accent

| Token pair | Word it carries |
| --- | --- |
| `text-success` on `bg-success-subtle` | Healthy, active, verified |
| `text-warning` on `bg-warning-subtle` | Near a limit, degraded, retrying |
| `text-danger` on `bg-danger-subtle` | Over budget, failed, revoked |
| `text-info` on `bg-info-subtle` | Read-only, informational, no action |
| `attention` | Action required, unread. **Not a fifth status** |

Light `--color-warning` is a bronze, not an amber. `text-warning` is real text in
23 places, so it has to clear 4.5:1, and at that hue the sRGB ceiling collapses as
lightness rises. If it reads dull, the fix is those call sites, not the token.

A status is never carried by color alone. Every mark ships a word beside it: a
red-green deficiency removes the only channel a bare dot has.

## Chart slots

`--color-chart-cat-1` through `cat-8` are categorical. **Assign in that order and
never cycle.** The color-deficiency margins are computed on adjacent pairs in
exactly that order, so wrapping around invalidates them. A ninth group folds into
`cat-other`, the neutral slot, which is the one exempt from the chroma floor
because "everything else" is not an identity.

`ramp-1` through `ramp-4` are one hue at four lightnesses, strongest first, for a
part-to-whole bar whose segments are ordered rather than named.

```tsx
// Correct
<Dot className="bg-chart-cat-1" />
<span className="text-danger">Over budget</span>

// Incorrect: a status hue is not a series color, and a hex is never a value
<Dot className="bg-[#08899f]" />
<span className="text-chart-cat-8">Over budget</span>
```
