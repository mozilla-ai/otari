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

**A component directory is named for the topic file that documents it**, so the
table below is the topic list with `@/shared/components/` in front of it. One
component per module, named for the file; where a module carries a second export
it is because the two are useless apart (a component and the lane it sits in, a
classifier and the meter that reads it).

| Module | Exports you will reach for |
| --- | --- |
| `@heroui/react` | `Button`, `Select`, `Tooltip`, `Spinner`, and the rest of HeroUI v3 |
| `layout/Section` · `/PageIntro` · `/SettingsGroup` · `/Toolbar` · `/TableScrollFrame` | one component each, named for its file |
| `metrics/KpiStrip` · `/KpiCell` · `/Meter` | one each |
| `metrics/SpendMeter` | `SpendMeter`, `spendState`, and the `SpendState` type |
| `metrics/SeverityMark` | `SeverityMark`, and the `Severity` type |
| `metrics/TrendChip` | `TrendChip`, `trendState`, and the `Trend*` types |
| `metrics/charts` | `TrendChart`, `Sparkline`, `ChartLegend`, and the `SeriesDef` / `StackedPoint` types |
| `feedback/ErrorBanner` · `/InfoBanner` · `/EmptyState` · `/EmptyMessage` · `/PageLoading` · `/ConfirmDialog` · `/errorMessage` | one each |
| `forms/Field` · `/SecretField` | `Field` · `SecretField` |
| `forms/FieldMessages` | `FieldMessages`, `ControlField` |
| `forms/Checkbox` | `Checkbox`, `CheckboxVisual` |
| `forms/inputClass` | `INPUT_CLASS` |
| `actions/ConfirmButton` · `/RefreshButton` · `/CopyButton` | one each |
| `actions/CopyField` | `CopyField`, `CopyableValue` |
| `actions/RowAction` | `RowAction`, `RowActionRow` |
| `actions/ConfirmRowAction` | `ConfirmRowAction` |
| `data/DataTable` | `DataTable`, and the `DataTableColumn` type |
| `data/TablePagination` | `TablePagination`, `PAGE_SIZE_OPTIONS` |
| `data/BulkActionBar` | `BulkActionBar` |
| `navigation/TabRow` | `TabRow`, `Tab` |
| `navigation/Segmented` · `/FilterSelect` · `/FilterMultiComboBox` | one each |
| `navigation/FilterChips` | `FilterChips`, and the `FilterChip` type |
| `indicators/Dot` · `/Badge` · `/DismissChip` | one each |
| `access/EntitlementGate` · `/UnavailableHere` · `/MissingGatewayAddressNotice` | one each |
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
  `web/src/shared/api/`, one module per domain (`usage.ts`, `apiKeys.ts`,
  `organizations.ts`, …). Never a raw `fetch`.
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
| [actions.md](actions.md) | The three button variants, sizes, places, icon-only, the two-step confirm, and the other action shapes (`RowAction`, `RefreshButton`, `CopyButton`) |
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

**Ours live in `shared/components/deprecated/`**, which is what makes this
mechanical rather than a rule somebody has to remember: the specifier says it at
the call site, and `deprecated/deprecated.test.ts` fails on a call site that is
not already in its list, naming what to use instead.

| Do not use | Use instead | Still in |
| --- | --- | --- |
| `deprecated/PageHeader` | `layout/PageIntro` | 4 pages, one use each |
| `deprecated/StatCard` | `metrics/KpiStrip` + `KpiCell` | Overview only, 4 uses |
| `deprecated/RowActions` | `actions/RowActionRow` | 1 use, in `PasskeysCard` |
| `deprecated/SettingsSection` | `layout/SettingsGroup` | **nothing. Dead code** |
| HeroUI `Card` | `Section`, or a bare band | 6 components |

The gate derives **which files** import each one, so a row naming a page that no
longer reaches for it fails. It does not count usages, so the "one use each" and
"4 uses" figures are prose and can drift the way the previous two did (`StatCard`
was listed on Usage after Usage stopped using it, and `RowActions` on two call
sites when it had one). Check them against the tree rather than against this
table.

`SettingsSection` is the row to act on: it has no call site anywhere, and it
shadowed `layout/SettingsGroup` while diverging from this tree's
`export function` convention. It is a deletion waiting for a maintainer rather
than a migration.

HeroUI `Card` is the one row with no module of ours behind it, so it stays a
review note rather than a gate.

Converting a remaining call site is welcome. Adding a new one is a review block.
