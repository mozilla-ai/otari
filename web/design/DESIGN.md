# Otari dashboard design system

One flat plane, partitioned by hairlines. No cards, no rounded corners, no shadow,
one teal accent spent once per screen. Density is the point: an operator reads
this dashboard to answer "what is this costing me and what is broken", so every
pixel of chrome is pixels not spent on data.

Read the topic file for what you are building before you build it. The rules below
apply everywhere and are the ones most often broken.

## The layer rule

Components read **semantic tokens only**. Never a raw value, never a primitive.

| Allowed | Banned | Why |
| --- | --- | --- |
| `bg-surface`, `text-muted`, `border-border` | `bg-[#ffffff]`, `bg-white`, `text-gray-500` | A hex at a call site cannot follow a theme, and there are two themes. |
| `text-caption`, `text-overline` | `text-[13px]`, `text-xs font-semibold uppercase` | A size spelled inline is a scale nobody can see the whole of. |
| `variant="ghost"` | `variant="outline"`, `variant="secondary"` | Three variants exist. A fourth compiles and renders unstyled. |

`src/styles/foundation.test.ts` fails the build on all three. It is not a style
preference, it is a gate.

## Four values that are single-sourced

- **Radius is `0px`.** One key in `globals.css` squares every corner in the app,
  HeroUI's internals included. Never add `rounded-*`.
- **Elevation is `none`.** Nothing stacks, so nothing casts. Divide with
  `border-border`, never with a shadow.
- **Emphasis is weight `550`.** The variable axis holds 400 and 550. Asking for
  600 renders 550 and hides the mistake.
- **Disabled is `opacity: 0.4`.** Set on the control by HeroUI. Never hand-rolled.

## Where things come from

Every example in these files omits its imports. Here they are once. The alias is
`@/`, mapped to `web/src/`, and every export is named. **There is no barrel:** import
from the module, not from `@/shared/components`.

| Module | Exports you will reach for |
| --- | --- |
| `@heroui/react` | `Button`, `Select`, `Tooltip`, `Spinner`, and the rest of HeroUI v3 |
| `@/shared/components/surface` | `Section`, `PageIntro`, `SettingsGroup`, `Toolbar`, `KpiStrip`, `KpiCell`, `SeverityMark`, `Meter`, `SpendMeter`, `spendState`, `Tab`, `TabRow`, `Segmented`, `RowAction`, `RowActionRow`, `ConfirmRowAction`, `DismissChip`, `Dot`, `EmptyMessage`, `TableScrollFrame` |
| `@/shared/components/ui` | `ConfirmButton`, `InfoBanner`, `ErrorBanner`, `EmptyState`, `PageLoading`, `Checkbox`, `CopyField`, `CopyButton`, `CopyableValue`, `RefreshButton`, `Badge`, `FilterSelect`, `FilterMultiComboBox`, `errorMessage` |
| `@/shared/components/DataTable` | `DataTable`, and the `DataTableColumn` type |
| `@/shared/components/Field` · `/SecretField` | `Field` · `SecretField` |
| `@/shared/components/TablePagination` | `TablePagination`, `PAGE_SIZE_OPTIONS` |
| `@/shared/components/BulkActionBar` | `BulkActionBar` |
| `@/shared/components/FilterChips` | `FilterChips`, and the `FilterChip` type |
| `@/shared/components/TrendChip` | `TrendChip`, `trendState` |
| `@/shared/components/charts` | `TrendChart`, `Sparkline`, `ChartLegend`, and the `SeriesDef` / `StackedPoint` types |
| `@/shared/components/ConfirmDialog` | `ConfirmDialog` |
| `@/features/settings/Toggle` | `Toggle` |
| `@/shared/helpers/format` | `formatUsd`, `formatUsdHeadline`, `formatNumber`, `formatTokens`, `formatPct`, `formatDate`, `formatDateTime`, `formatRelative`, `deltaFraction` |

**Never hand-roll a formatter.** `toLocaleString()` at a call site is how two pages
come to print the same number differently. Every value in a cell or a `KpiCell` goes
through `@/shared/helpers/format`.

A page component is a named export matching its filename. Icons come from
`react-icons/fi`, sized `size-4` (`h-3.5 w-3.5` inside a `sm` button), always
`aria-hidden`.

## What these docs do not cover

How a screen gets its data. A guess there will be wrong:

- Fetching, caching, mutations: the TanStack Query hooks in
  `web/src/shared/api/hooks.ts`. Never a raw `fetch`.
- Filter and page state that must survive a reload: `web/src/shared/helpers/urlState.ts`,
  not `useState`.
- Routing, and adding a rail destination: `web/src/routes/` and the nav registry.
- Whether to memoize: do not. The React Compiler is enabled.

All four live in [frontend-standards](../../.github/skills/frontend-standards/SKILL.md)
and [web/AGENTS.md](../AGENTS.md).

## Topics

| File | Covers |
| --- | --- |
| [colors.md](colors.md) | Surfaces, text ramp, borders, the accent's five jobs, status, chart slots |
| [typography.md](typography.md) | The 12 type roles, the ladder rule, the three families |
| [layout.md](layout.md) | Bands, the bleed rule, `Section`, `PageIntro`, `SettingsGroup`, page recipes |
| [buttons.md](buttons.md) | The three variants, sizes, places, icon-only, two-step confirm |
| [forms.md](forms.md) | `Field`, `SecretField`, `Toggle`, `Checkbox`, selects, validation timing |
| [data.md](data.md) | `DataTable`, pagination, bulk actions |
| [metrics.md](metrics.md) | KPI strip, trends, meters, status marks, charts |
| [feedback.md](feedback.md) | Banners, empty states, loading, dialogs |
| [navigation.md](navigation.md) | `TabRow`, `Segmented`, `FilterChips`, the rail |
| [motion-and-access.md](motion-and-access.md) | Durations, press, focus, 44px floor, reduced motion |

Visual reference: the Paper file, `Otari / Neat shell`, has the same system as
artboards (foundations, components with every state, page archetypes).

## Components that exist but must not be used in new code

The redesign is not uniformly converted. These are still in the tree with live
call sites, and reaching for one puts a card back on a flat page.

| Do not use | Use instead | Still in |
| --- | --- | --- |
| `PageHeader` (`shared/components/ui.tsx`) | `PageIntro` (`shared/components/surface.tsx`) | 4 pages |
| `StatCard` (`shared/components/ui.tsx`) | `KpiStrip` + `KpiCell` | Overview, Usage |
| HeroUI `Card` | `Section`, or a bare band | 6 components |

Converting a remaining call site is welcome. Adding a new one is a review block.
