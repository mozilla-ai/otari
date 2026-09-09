# Metrics

The numbers a page leads with, and the marks that qualify them. Tables are in
[data.md](data.md).

```
Is it a handful of headline numbers for the whole page?
 └── KpiStrip + KpiCell            (never StatCard; see DESIGN.md)
Is it one number's change over a period?
 └── TrendChip                      (inside a KpiCell's meta line, or a table cell)
Is it a value against a limit?
 └── SpendMeter (money against a ceiling) or Meter (a bare fraction)
Is it a status?
 └── SeverityMark, which ships the word beside the mark
Is it a quantity over time?
 └── TrendChart, or Sparkline when it sits inside a cell
Is it a part-to-whole with ordered segments?
 └── A stacked bar on ramp-1..4
```

## KpiStrip and KpiCell

```ts
KpiStrip: { empty: boolean, children }
KpiCell: { label: string, value: string, severity?: Severity, subline?: string,
  delta?: ReactNode, graphic?: ReactNode }
Severity: { status: "ok" | "warn" | "alert", word: string }
```

`empty` is required and means **this page has no data in the selected range**, not
"this cell has no value". It is the page's own emptiness test, passed down so the
strip can shorten as one: it drops the reserved graphic row from every cell at once.
Pass it together with passing no `graphic` to any cell; passing one without the other
is what leaves one cell taller than its neighbors.

`value` is a formatted string, from `@/shared/helpers/format` and never from
`toLocaleString()` at the call site. `delta` and `graphic` take elements (a
`TrendChip`, a `Sparkline`); `severity` takes data.

A cell may carry `severity`, `delta` and `subline` at once: the meta line drops the
subline first, because its facts also live in the breakdown table below while a delta
lives nowhere else.

**Five cells.** The strip's grid is a fixed 2 / 3 / 5 columns by breakpoint, so four
cells leave one empty track on a wide viewport and three leave two. Both call sites
in the product pass five. If a page has fewer than five things worth leading with,
it does not want this band: put the one number in the `PageIntro` sentence instead.

A cell has four rows and **always four**, in this order:

1. `label`: overline, bottom-aligned so a wrapped label still shares its baseline.
2. `value`: `text-mono-figure`, weight 400, tabular.
3. The meta line: `severity`, then `delta`, then `subline`, most-judged to least.
   One line, never two, and only the subline truncates.
4. `graphic`: a `Sparkline`. Reserved even when absent.

The reserved rows are the whole point: an absent sparkline must not make one cell
shorter than the four beside it.

Always pass `subline`, in every page state. An em dash with nothing under it makes
the reader guess whether the number is missing or zero.

## TrendChip

```ts
TrendChip: { fraction: number | null | undefined, polarity?: TrendPolarity,
  size?: "sm" | "md" | "lg", text?, caption?, className? }
TrendPolarity: "up-is-good" | "down-is-good" | "neutral"   // default "neutral"
```

`fraction` is a **ratio, not a percentage**: `0.124` prints "+12.4%". `deltaFraction`
in `@/shared/helpers/format` computes one from a current and prior value.

`polarity` is a prop, not a color choice: `up-is-good`, `down-is-good`, `neutral`.
A rising cost is not good news. The call site declares which direction is good and
the chip picks its own hue.

```tsx
// Correct
<TrendChip fraction={delta} polarity="down-is-good" size="md" caption="vs prior 30 days" />

// Incorrect: hardcoding the hue is how a rising spend turns green
<span className="text-success">▲ {formatPct(delta)}</span>
```

`fraction` of `null` or `undefined` renders nothing, so a caller never branches.

## Meters

```ts
SpendMeter: { spent, allocated, ariaLabel, nearLimitAt = 0.8, className? }
Meter: { fraction, ariaLabel }
SeverityMark: { severity: { status: "ok" | "warn" | "alert", word: string } }
Dot: { className }
```

`SeverityMark` is the status mark **with its word**, and it is what a status belongs
in: `Dot` is the 6px square alone, for a cell that supplies its own word beside it.
Never hand-roll `<span className="text-success">Active</span>`; that is a status
whose only channel is hue.

`SpendMeter` takes `spent` and `allocated` and derives its own state:
`on-track` (accent), `near-limit` (warning, from `nearLimitAt`, default 0.8),
`over` (danger, bar full rather than overflowing). Never compute the color at the
call site; use `spendState` if you need the word.

## Charts

Data ink is the accent and the chart slots. No gridline is heavier than a hairline
and no chart sits in a box.

- `TrendChart` for a series over time. Pass `ariaLabel` and `formatValue`; both are
  required because a chart with no accessible name is a picture.
- `Sparkline` inside a cell, 32px tall by default.
- `ChartLegend` renders nothing below two series, which is correct: a legend for one
  series is decoration.
- Meters and spinners stay on `--color-primary`, and so do charts. They are quantity
  and motion, not state, so they do **not** take `control-indicator`.
