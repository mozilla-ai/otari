# Otari dashboard design system

One flat plane, partitioned by hairlines. No cards, no rounded corners, no shadow,
one teal accent spent once per screen. Density is the point: an operator reads
this dashboard to answer "what is this costing me and what is broken", so every
pixel of chrome is pixels not spent on data.

Read the topic file for what you are building before you build it. The rules below
apply throughout the dashboard, with the scoped
[public authentication exception](layout.md#public-authentication) below.

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
from the module, not from `@/design-system`.

**A component directory is named for the topic file that documents it**, so the
table below is the topic list with `@/design-system/` in front of it. One
component per module, named for the file; where a module carries a second export
it is because the two are useless apart (a component and the lane it sits in, a
classifier and the meter that reads it).

Two directories are the exception, and both sit outside the design system rather
than inside it: `@/shared/components/access/` and
`@/shared/components/deprecated/`. See "The extraction contract" below for why.

Three directory names inside it are coined rather than borrowed from a topic
file, so nobody mistakes them for the system's own vocabulary: `indicators/` and
`access/` were named in [module-map.md](module-map.md), and `content/` is named
here, for rendered prose. `helpers/` and `hooks/` are the layer's internals
(`clipboard`, the two display formatters, `useConfirmationFocus`); a page reaches
for a component, not for those.

| Module | Exports you will reach for |
| --- | --- |
| `@heroui/react` | `Button`, `Select`, `Tooltip`, `Spinner`, and the rest of HeroUI v3 |
| `layout/Section` · `/PageIntro` · `/SettingsGroup` · `/SettingRow` · `/Toolbar` · `/TableScrollFrame` | one component each, named for its file |
| `metrics/KpiStrip` · `/KpiCell` · `/Meter` · `/ProgressBar` | one each |
| `metrics/SpendMeter` | `SpendMeter`, `spendState`, and the `SpendState` type |
| `metrics/SeverityMark` | `SeverityMark`, and the `Severity` type |
| `metrics/TrendChip` | `TrendChip`, `trendState`, and the `Trend*` types |
| `metrics/charts` | `TrendChart`, `Sparkline`, `ChartLegend`, and the `SeriesDef` / `StackedPoint` types |
| `feedback/ErrorBanner` · `/InfoBanner` · `/EmptyState` · `/EmptyMessage` · `/PageLoading` · `/PageError` · `/ConfirmDialog` · `/FormDialog` · `/ScanBorder` · `/ErrorBoundary` · `/Skeleton` · `/errorMessage` | one each |
| `feedback/Dialog` | `Dialog`, `DialogSection`, and the `DialogSize` type |
| `forms/Field` · `/SecretField` · `/TextArea` · `/SearchField` · `/FieldAction` | one component each |
| `forms/Select` | `Select`, and the `SelectOption` type |
| `forms/ComboBoxField` | `ComboBoxField`, and the `ComboBoxOption` type |
| `forms/MultiSelect` | `MultiSelect`, and the `MultiSelectOption` type |
| `forms/ComboBoxEmpty` | `ComboBoxEmpty`. The two sentences an empty popover picks between |
| `forms/RadioGroup` | `RadioGroup`, and the `RadioOption` type |
| `forms/Toggle` | `Toggle` |
| `forms/FieldMessages` | `FieldMessages`, `ControlField` |
| `forms/Checkbox` | `Checkbox`, `CheckboxVisual` |
| `forms/inputClass` | `INPUT_CLASS` |
| `forms/optionKey` | `optionKey`, `optionValue`. Internal to the two selects |
| `actions/Button` | `Button`, and the `ButtonVariant` / `ButtonSize` / `ButtonProps` types |
| `actions/IconButton` | `IconButton` |
| `actions/ConfirmButton` · `/RefreshButton` · `/CopyButton` | one each |
| `actions/CopyField` | `CopyField`, `CopyableValue` |
| `actions/RowAction` | `RowAction`, `RowActionRow` |
| `actions/ConfirmRowAction` | `ConfirmRowAction` |
| `data/DataTable` | `DataTable`, and the `DataTableColumn` type |
| `data/TablePagination` | `TablePagination`, `PAGE_SIZE_OPTIONS` |
| `data/BulkActionBar` | `BulkActionBar` |
| `navigation/TabRow` | `TabRow`, `Tab` |
| `navigation/Segmented` · `/FilterSelect` · `/FilterMultiComboBox` · `/Disclosure` · `/DisclosureRow` · `/DocsLink` | one each |
| `navigation/FilterChips` | `FilterChips`, and the `FilterChip` type |
| `overlays/Tooltip` · `/Popover` | one each |
| `indicators/Dot` · `/Badge` · `/DismissChip` · `/Kbd` · `/Avatar` | one each |
| `indicators/Chip` | `Chip`, and the `ChipTone` type |
| `layout/Divider` | `Divider` |
| `layout/ListDetail` | `ListDetail`, `ListDetailRow`. See [layout.md](layout.md) |
| `content/Markdown` | `Markdown` |
| `content/CodeBlock` | `CodeBlock`. Two arrangements: a language row, or bare with a floating copy |
| `ProductMark` | `ProductMark`. At the top level: it belongs to no topic |
| `@/shared/components/access/EntitlementGate` · `/UnavailableHere` · `/MissingGatewayAddressNotice` | one each |
| `@/shared/helpers/format` | `formatUsd`, `formatUsdHeadline`, `formatNumber`, `formatTokens`, `formatPct`, `formatDate`, `formatDateTime`, `formatRelative`, `deltaFraction` |

**Never hand-roll a formatter.** `toLocaleString()` at a call site is how two pages
come to print the same number differently. Every value in a cell or a `KpiCell` goes
through `@/shared/helpers/format`.

A page component is a named export matching its filename. Icons come from
`react-icons/fi`, sized `size-4` (`h-3.5 w-3.5` inside a `sm` button), always
`aria-hidden`.

## The extraction contract

`src/design-system/` is a library that happens to live in this repository. It is
not a package yet, and the point of the contract is that turning it into one
stays a folder move rather than an untangling.

**The rule: it may import nothing under `src/`.** React, HeroUI,
react-aria-components, react-icons, recharts, react-markdown, and its own modules. That is the
whole allowance, and `biome.jsonc` rejects the rest with
`src/architecture.test.ts` proving each rejection. The question it answers is not
"does this import point the wrong way" but **"would this directory still compile
with the rest of `src/` deleted"**.

Which is why two directories stayed behind in `shared/components/`:

| Stayed | Why it is not a primitive |
| --- | --- |
| `access/EntitlementGate` · `/UnavailableHere` · `/MissingGatewayAddressNotice` | "Unavailable here" is a fact about which deployment answered the page. A surface and a capability are this product's axes, and a design system does not know deployments exist. |
| `deprecated/*` | It is on its way out. Importing four components into a fresh library on the day it is created is not a migration, it is a starting position nobody would choose. |

Everything else moved, including two things that look like they should not have:

- **`errorMessage`** named `ApiError` and so imported the transport. It tested
  `ApiError` before `Error` and returned `error.message` from both, and
  `ApiError extends Error`, so the first branch could never change an answer.
  Deleting the dead branch was the whole cost of moving it, and `ErrorBanner`
  came with it.
- **`formatPct` and `formatRelative`** live in `design-system/helpers/format`
  because `TrendChip` and `RefreshButton` need them. `@/shared/helpers/format`
  is still the module a page reaches for and still the only one named above: it
  re-exports those two, so there is one implementation and no call site moved.
  The rest of it (`formatUsd`, `formatTokens`, `formatCost`, `formatContext`)
  stayed, because a spend figure and a token count are this product's vocabulary
  rather than a design system's.

**What a component here may not do.** It is presentational and stateless: no
TanStack Query, no `useDeployment`, no router, no context of the app's. State is
a prop and a callback, which is why every control is controlled and why `Avatar`
takes `initials` rather than a name. A component that needs the transport, a
domain formatter or a generated API type is an *application* component: it
belongs in `src/shared/components/`, composing the primitive.

**What is deliberately not here.** Breadth was the goal, but a component with no
consumer is an API to maintain and a lie about what the product uses, so five
plausible primitives were declined and are worth naming so the next person does
not have to re-derive it: a `Spinner` wrapper (HeroUI's is used directly in eight
places and a wrapper adds nothing), a generic `Alert` (`ErrorBanner` and
`InfoBanner` already are it), `Stack` (it would fight the `gap-*` on the parent
rule that responsiveness.md sets), a `Link` (HeroUI's is a full page reload for
an internal route, so a wrapper would be a trap, and the working answer is
TanStack's), and `Breadcrumbs` (`app/nav/Breadcrumbs` is chrome and reads the
router). Ask for one when something needs it.

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
| [layout.md](layout.md) | Bands, the bleed rule, `Section`, `PageIntro`, `SettingsGroup`, `SettingRow`, page recipes |
| [actions.md](actions.md) | The three button variants, sizes, places, icon-only, the delete confirm dialog and the two-step confirm that is left beside it, and the other action shapes (`RowAction`, `RefreshButton`, `CopyButton`) |
| [forms.md](forms.md) | `Field`, `SecretField`, `Toggle`, `Checkbox`, selects, validation timing |
| [data.md](data.md) | `DataTable`, pagination, bulk actions |
| [metrics.md](metrics.md) | KPI strip, trends, meters, status marks, charts |
| [feedback.md](feedback.md) | Banners, empty states, loading, dialogs |
| [navigation.md](navigation.md) | `TabRow`, `Segmented`, `FilterChips`, the rail |
| [overlays.md](overlays.md) | `Tooltip`, `Popover`, the two dialogs, and which of them a thing wants |
| [motion-and-access.md](motion-and-access.md) | Durations, press, focus, 44px floor, reduced motion |

Visual reference, in two places. The Paper file `Otari / Neat shell` has the
same system as artboards (foundations, components with every state, page
archetypes). The **component catalog** has it as running code: every component
here with its variants, its states and both themes, one story per axis. Run it
with `pnpm --dir web run storybook`; `.github/workflows/otari-design-system.yml`
publishes it from main and gates a pull request on it building. Rendering every
story is a separate sweep, on main and on demand, and it is worth running
locally before pushing. See [web/.storybook/README.md](../.storybook/README.md)
for both and for why they are split.

## Components that exist but must not be used in new code

The redesign is not uniformly converted. These are still in the tree with live
call sites, and reaching for one puts a card back on a flat page.

**Ours live in `shared/components/deprecated/`**, which is what makes this
mechanical rather than a rule somebody has to remember: the specifier says it at
the call site, and `deprecated/deprecated.test.ts` fails on a call site that is
not already in its list, naming what to use instead.

| Do not use | Use instead | Still in |
| --- | --- | --- |
| `deprecated/StatCard` | `metrics/KpiStrip` + `KpiCell` | **nothing. Dead code** |
| `deprecated/RowActions` | `actions/RowActionRow` | 1 use, in `PasskeysCard` |
| `deprecated/SettingsSection` | `layout/SettingsGroup` | **nothing. Dead code** |
| HeroUI `Card` | `Section`, or a bare band | 6 components |

The gate derives **which files** import each one, so a row naming a page that no
longer reaches for it fails. It does not count usages, so the "one use each"
figures are prose and can drift the way the previous two did (`StatCard`
was listed on Usage after Usage stopped using it, and `RowActions` on two call
sites when it had one). Check them against the tree rather than against this
table.

`SettingsSection` and `StatCard` are the rows to act on: neither has a call
site anywhere. `SettingsSection` also shadowed `layout/SettingsGroup` while
diverging from this tree's `export function` convention. Both are deletions
waiting for a maintainer rather than migrations.

HeroUI `Card` is the one row with no module of ours behind it, so it stays a
review note rather than a gate.

Converting a remaining call site is welcome. Adding a new one is a review block.
