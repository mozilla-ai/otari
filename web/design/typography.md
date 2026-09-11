# Typography

Pick a role by what the text **means**, never by how big you want it. Twelve roles
cover the product. A size spelled inline (`text-[13px]`, or `text-xs font-semibold
uppercase`) is a scale nobody can see the whole of, and `foundation.test.ts` fails
on it.

## Three families

| Token | Family | For |
| --- | --- | --- |
| `--font-display` | Mozilla Headline | Page titles and one announcement head. Nothing else |
| `--font-sans` | Mozilla Text | Everything else. Two weights on the axis: 400 and 550 |
| `--font-mono` | Fira Code | Figures, keys, ids, code. Tabular numerals under a counting value |

## The roles

| Utility | Metrics | Use for |
| --- | --- | --- |
| `text-display` | 28/34, 550, display | The page's name. One per route |
| `text-display-sub` | 20/28, 550, display | A page's announcement moment: a get-started strip, a first-run panel |
| `text-heading` | 18/26, 550, sans | A section head. Sans, not the slab: it sits under the title |
| `text-title` | 16/24, 550 | A panel, a dialog, a group of fields |
| `text-body` | 14/20, 400 | The default. Inherited from `body`, so you rarely spell it |
| `text-emphasis` | 14/20, 550 | Emphasized body. Use rarely |
| `text-caption` | 13/19, 400, muted | A secondary line under a value |
| `text-subtle` | **not a type role** | A color, applied *with* a role: `text-caption text-subtle`. See [colors.md](colors.md) |
| `text-overline` | 12/18, 550, 0.06em, uppercase, subtle | A label above a group, and every table column head |
| `text-mono-figure` | 30/36, 400, tabular | A KPI figure |
| `text-mono-title` | 15px | A key name inline in a 16px heading |
| `text-mono-caption` | 13px | Model ids and counts in a row |
| `text-mono-micro` | 11px | Axis ticks, a truncated id |
| `text-mono-overline` | 11px, 0.1em, uppercase | A mono label above a group |
| `text-shell-label` | 13/19 | A rail row, a menu row, the top bar |
| `text-shell-secondary` | 12/18 | The second line under one of those |

The shell reads one step below the content scale on purpose: it is furniture around
a page, not part of one. The three `shell-*` roles carry metrics only, no color and
no weight, because one menu row is `text-foreground` resting and `text-muted`
disabled.

## Which role?

```text
Is it the name of the route?
 ├── Yes -> text-display                (one per page, nothing on the page is larger)
 └── No
      ├── Is it the page's one announcement (get started, first run)?
      │    └── Yes -> text-display-sub
      ├── Does it head a band or a section of the page?
      │    └── Yes -> text-heading
      ├── Does it title a panel, a dialog, or a group of fields?
      │    └── Yes -> text-title
      ├── Is it a label above a group, or a table column head?
      │    └── Yes -> text-overline
      ├── Is it a number read as a quantity?
      │    └── Yes -> text-mono-figure (a KPI) or text-mono-caption (in a row)
      ├── Is it explaining the value next to it?
      │    └── Yes -> text-caption
      └── Default -> text-body, which you inherit. Write nothing.
```

## The ladder rule

**Nothing inside a page may be larger than the page's own name.** The 30px KPI
figure is mono at weight 400; the 28px page title is display at 550. That is why
they do not fight, and it is why a figure never goes semibold: at 30px the size is
already the hierarchy.

Two documented off-scale values exist, both optical corrections across families:
`text-mono-title` at 15px (mono at 16 looms beside sans at 16) and the 9px avatar
monogram (two letters in a 26px circle are recognized, not read). The bar for a
third is that the value does something a scale step cannot. "No step happens to be
this size" is not a reason.

```tsx
// Correct
<h1 className="text-display">API keys</h1>
<span className="text-overline">Total spend</span>
<span className="text-mono-figure">{formatUsd(total)}</span>

// Incorrect: a hand-spelled scale, and weight 600 does not exist on the axis
<h1 className="text-[28px] leading-[34px] font-semibold">API keys</h1>
<span className="text-xs font-bold uppercase tracking-wider">Total spend</span>
```
