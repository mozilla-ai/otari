# Module map: giving `shared/` a shape

**This is a migration record, not a design topic.** It is deliberately absent
from DESIGN.md's topics table, because it describes work rather than a rule.
Delete it once the follow-ups at the bottom are closed; the durable half (where a
component lives) is already in DESIGN.md's "Where things come from" table.

**Superseded, in one respect: every path below that reads
`shared/components/<topic>/` is now `design-system/<topic>/`.** The ten topic
directories this change created were moved wholesale into a layer that may
import nothing else under `src/`, which is what makes them extractable as a
package; DESIGN.md's "The extraction contract" is the current rule, and the two
directories that stayed behind (`access/`, `deprecated/`) are named there. The
paths are left as written here rather than rewritten, because this file records
what a past change did and a record edited to match the present stops being one.
Read it for the reasoning, not for a location.

**Status: executed.** `surface.tsx` (876 lines, 22 exports), `ui.tsx` (1,075
lines, 20 exports) and `shared/api/hooks.ts` (3,638 lines, 172 exports) are gone,
replaced by 10 topic directories under `shared/components/` and 17 domain modules
under `shared/api/`. Verified with `lint`, `typecheck`, 135 test files / 3,844
tests, and a production build. Several things came out differently from the plan
below; each is marked **revised** where it applies.

## What was wrong

`shared/components/` was a flat directory of 40 entries in which two files held 42
exports between them:

| File | Lines | Exports | Call sites rewritten |
| --- | --- | --- | --- |
| `ui.tsx` | 1,075 | 20 | 60 files |
| `surface.tsx` | 876 | 22 | 48 files |

Neither name says anything. `ui` names the whole layer it sits in. `surface` is
worse than uninformative: `--color-surface` and `bg-surface` are a *token* meaning
one of four neutral levels, so the same word carries two unrelated meanings in one
tree, and neither reading predicts that `SpendMeter` and `Tab` are in the file.

Three pieces of evidence that the split is already half-decided:

- **`ui.tsx` and a `ui/` directory coexist.** `ui/RowActions.tsx` and
  `ui/SettingsSection.tsx` are already separate, so `@/shared/components/ui` and
  `@/shared/components/ui/RowActions` are both live specifiers.
- **`surface.tsx` has no `surface.test.tsx`.** Its coverage is already eleven
  per-component files (`section`, `pageIntro`, `settingsGroup`, `kpiCell`,
  `spendMeter`, `tabs`, `rowAction`, `dismissChip`, `tableScrollFrame`,
  `confirmRowActionFocus`, `dotRamp`), each importing `./surface`. The components
  were treated as separate before the source was.
- **[actions.md](actions.md) (then `buttons.md`) documented one family living in three modules.**
  `RowAction`, `RowActionRow`, `RowActions`, `ConfirmButton`, `ConfirmRowAction`,
  `RefreshButton` and `CopyButton` are one table there, and are spread across
  `surface.tsx`, `ui.tsx` and `ui/RowActions.tsx` here.

## The naming principle

**A directory takes the name of the design topic that documents it.** The design
system already partitions this vocabulary, the team and the agents already read
those files before writing UI, and mirroring them makes the doc-to-code mapping
one-to-one. That is the same reason AGENTS.md gives for keeping a fact in one
layer: a taxonomy invented here would be a second naming of something
`web/design/` already names, and the second copy is the one that goes stale.

Six of the ten topics are component groups and give their directories their names.
Two directories have no topic behind them and are named here for the first time;
they are called out below so nobody mistakes them for the system's own vocabulary.

## Target tree

```text
shared/components/
├── layout/       ← design/layout.md
├── metrics/      ← design/metrics.md
├── feedback/     ← design/feedback.md
├── forms/        ← design/forms.md
├── actions/      ← design/actions.md (renamed from buttons.md)
├── data/         ← design/data.md
├── navigation/   ← design/navigation.md
├── indicators/   coined here: the status marks from design/colors.md
├── access/       coined here: deployment and entitlement states, no topic
├── deprecated/   the review-blocked set, made mechanical
└── ProductMark.tsx
```

`ProductMark` stays at the top level on purpose. It is the product mark rather than
an icon (see web/AGENTS.md on why it is inline SVG at all), it belongs to no topic,
and a directory of one file is worse than a file.

### `buttons.md` became `actions.md`

The directory is `actions/`, not `buttons/`, and the topic file was renamed to
match. `CopyField` and `RowAction` are not buttons, that file's own closing section
was already titled "Other action shapes", and the three-variant rule it opens with
is about `Button` specifically. Renaming the doc was cheaper than a directory whose
name is contradicted by two thirds of its contents.

### The two coined names

`indicators/` and `access/` have no topic file, so they are the one place this spec
adds vocabulary rather than borrowing it:

- `indicators/` holds the small status marks (`Dot`, `Badge`, `DismissChip`).
  `colors.md` describes the status roles they wear but documents no component.
- `access/` holds what renders in place of a page the deployment does not serve or
  the caller is not entitled to (`EntitlementGate`, `UnavailableHere`,
  `MissingGatewayAddressNotice`). This is the surface and entitlement axes reaching
  the UI, described in web/AGENTS.md and ARCHITECTURE.md rather than in a design
  topic, because it is a deployment fact and not a visual one.

If either name is unwanted, the fallback is `indicators/` into `metrics/` and
`access/` into `feedback/`. Neither fallback is recommended: an entitlement panel is
not feedback about an action, and a brand badge is not a metric.

## Move table

Every export in the two files, and every loose file. Tests move with their subject.

### From `surface.tsx`

| Export | Destination |
| --- | --- |
| `Section` | `layout/Section.tsx` |
| `PageIntro` | `layout/PageIntro.tsx` |
| `SettingsGroup` | `layout/SettingsGroup.tsx` |
| `Toolbar` | `layout/Toolbar.tsx` |
| `TableScrollFrame` | `layout/TableScrollFrame.tsx` |
| `KpiStrip` | `metrics/KpiStrip.tsx` |
| `KpiCell` | `metrics/KpiCell.tsx` |
| `Meter` | `metrics/Meter.tsx` |
| `SpendMeter`, `spendState`, `SpendState` | `metrics/SpendMeter.tsx` |
| `SeverityMark`, `Severity` | `metrics/SeverityMark.tsx` |
| `Tab`, `TabRow` | `navigation/TabRow.tsx` |
| `Segmented` | `navigation/Segmented.tsx` |
| `RowAction`, `RowActionRow` | `actions/RowAction.tsx` |
| `ConfirmRowAction` | `actions/ConfirmRowAction.tsx` |
| `EmptyMessage` | `feedback/EmptyMessage.tsx` |
| `Dot` | `indicators/Dot.tsx` |
| `DismissChip` | `indicators/DismissChip.tsx` |

`TableScrollFrame` goes to `layout/` rather than `data/` because layout.md lists it
among the band components. `Tab` ships with `TabRow` and `RowActionRow` with
`RowAction`: each pair is a component and the lane it sits in, useless apart.

### From `ui.tsx`

| Export | Destination |
| --- | --- |
| `Checkbox`, `CheckboxVisual` | `forms/Checkbox.tsx` |
| `INPUT_CLASS` | `forms/inputClass.ts` |
| `ErrorBanner` | `feedback/ErrorBanner.tsx` |
| `InfoBanner` | `feedback/InfoBanner.tsx` |
| `EmptyState` | `feedback/EmptyState.tsx` |
| `PageLoading` | `feedback/PageLoading.tsx` |
| `errorMessage` | `feedback/errorMessage.ts` |
| `ConfirmButton` | `actions/ConfirmButton.tsx` |
| `RefreshButton` | `actions/RefreshButton.tsx` |
| `CopyButton` | `actions/CopyButton.tsx` |
| `CopyField`, `CopyableValue` | `actions/CopyField.tsx` |
| `FilterSelect` | `navigation/FilterSelect.tsx` |
| `FilterMultiComboBox` | `navigation/FilterMultiComboBox.tsx` |
| `Badge` | `indicators/Badge.tsx` |
| `UnavailableHere` | `access/UnavailableHere.tsx` |
| `PageHeader` | `deprecated/PageHeader.tsx` |
| `StatCard`, `StatStatus` | `deprecated/StatCard.tsx` |

`errorMessage` and `INPUT_CLASS` take `.ts`, not `.tsx`: naming-conventions.md
spells a pure module `camelCase.ts`, and neither returns JSX.

### Loose files (moved whole, contents untouched)

| File | Destination |
| --- | --- |
| `DataTable.tsx` | `data/DataTable.tsx` |
| `TablePagination.tsx` | `data/TablePagination.tsx` |
| `BulkActionBar.tsx` | `data/BulkActionBar.tsx` |
| `Field.tsx` | `forms/Field.tsx` |
| `SecretField.tsx` | `forms/SecretField.tsx` |
| `FieldMessages.tsx` | `forms/FieldMessages.tsx` |
| `charts.tsx` | `metrics/charts.tsx` |
| `TrendChip.tsx` | `metrics/TrendChip.tsx` |
| `ConfirmDialog.tsx` | `feedback/ConfirmDialog.tsx` |
| `FilterChips.tsx` | `navigation/FilterChips.tsx` |
| `EntitlementGate.tsx` | `access/EntitlementGate.tsx` |
| `MissingGatewayAddressNotice.tsx` | `access/MissingGatewayAddressNotice.tsx` |
| `ui/RowActions.tsx` | `deprecated/RowActions.tsx` |
| `ui/SettingsSection.tsx` | `deprecated/SettingsSection.tsx` |
| `ProductMark.tsx` | unchanged |

### Test renames that come free

The eleven camelCase test files exist because they had no 1:1 subject to be named
after. naming-conventions.md asks for `<Name>.test.tsx`, so each takes its
subject's name on the way:

`section` → `layout/Section`, `pageIntro` → `layout/PageIntro`, `settingsGroup` →
`layout/SettingsGroup`, `tableScrollFrame` → `layout/TableScrollFrame`, `kpiCell` →
`metrics/KpiCell`, `spendMeter` → `metrics/SpendMeter`, `dotRamp` →
`indicators/Dot`, `dismissChip` → `indicators/DismissChip`, `tabs` →
`navigation/TabRow`, `rowAction` → `actions/RowAction`, `confirmRowActionFocus` →
`actions/ConfirmRowAction`.

`ui.test.tsx` covers exports that end up in five directories, and its eight
top-level describes are one component each with no shared prose, so it split
cleanly into eight files.

**Revised, two of them.** `tabs.test.tsx` stayed whole, as
`navigation/tabs.test.tsx`: it covers `Tab`, `TabRow` and `Segmented` under one
header docstring arguing all three ARIA departures together, and splitting it
would have meant duplicating that rationale across two files. It keeps a
camelCase name because it has no single subject, which is the same reason the
eleven had camelCase names to begin with.

`dotRamp.test.ts` did not go to `indicators/` either, because it is not a
component test: it is a source-scanning gate over the whole tree (row actions
never paint danger at rest; status dots never take a surface-ramp value), so it
went to `src/styles/` beside `foundation.test.ts` and `monoScale.test.ts`. Two
things had to move with it. Its root is now `join(process.cwd(), "src")`, the
spelling its new neighbors use, and it gained the "covers the source tree" guard
they have, because both its rules assert only that an offender list is empty and
would have passed over nothing at all. And it carried a hardcoded
`file.endsWith("surface.tsx")` exception for `RowAction`'s own definition, which
the split broke; it now names `actions/RowAction.tsx`. That exception, not a path
bug, is what made the suite go red mid-refactor.

## `deprecated/` is the point, not a parking space

DESIGN.md keeps a prose table of components that must not be used in new code, and
"adding a new one is a review block" is enforced today by a human noticing. A
directory makes it mechanical: one `noRestrictedImports` pattern for
`@/shared/components/deprecated/**` scoped to everything outside it, or a
`foundation.test.ts` assertion that the import count per module never rises.

Four members, two of which this spec is the first to name:

| Component | Call sites | Replacement | Why |
| --- | --- | --- | --- |
| `PageHeader` | 4 pages | `layout/PageIntro` | Already in DESIGN.md's table |
| `StatCard` | 2 pages | `metrics/KpiStrip` + `KpiCell` | Already in DESIGN.md's table |
| `RowActions` | 1 | `actions/RowActionRow` | actions.md already says "do not reach for it", and puts it on two |
| `SettingsSection` | **0** | `layout/SettingsGroup` | Undocumented near-duplicate, and an `export const` arrow against this tree's `export function` |

HeroUI `Card` is the third row of DESIGN.md's table and is not here, because it is a
library component rather than one of ours; it stays a review note.

## `shared/api/hooks.ts`

A different problem, and "one file per hook" is the wrong answer to it. The file is
3,638 lines and 172 exports, but it is also 60 query-key constants, 109
`invalidateQueries` calls across 39 distinct keys, and 14 shared helpers. Splitting
per hook would be 172 files and would scatter the invalidation graph, which is
exactly what that file's comments exist to protect (`DISCOVERABLE` deliberately not
nested under `MODELS`; `ORGANIZATION_BUDGETS` keyed apart from `BUDGETS`).

One module per domain, and `hooks` dropped from the path: under `api/` every export
is already a hook, so the segment names nothing.

```text
shared/api/
  client.ts        unchanged
  queryKeys.ts     all 60 constants. MUST stay one module
  paging.ts        fetchAllPaged and the page caps
  apiKeys.ts       useKeys, useCreateKey, useRotateKey, …
  usage.ts · organizations.ts · workspaces.ts · providers.ts
  routing.ts · budgets.ts · pricing.ts · models.ts · tools.ts
  account.ts · settings.ts · deployment.ts
```

Spelled `apiKeys.ts` rather than `keys.ts` so it does not read as a sibling of
`queryKeys.ts`. The key constants stay in one module because cross-domain
invalidation is the file's central fact, not an accident: a key per domain module
would either duplicate constants or reintroduce the import cycle the no-barrel rule
exists to prevent.

## No barrels, so every call site changes

imports-and-modules.md forbids an `index.ts` that re-exports its siblings, with
reasons (bundler pull-in, `index.ts` tab soup, hidden stack-trace locations,
circular imports). Only three `index` files exist and each has a stated job, so
there is no way to keep call sites stable: all 61, 35 and 59 importing files are
rewritten. That is the cost of the rule and it is worth paying; a barrel here would
be the fourth exception and the first one without a job.

## What the executing change also has to touch

Moving the files is the easy half.

- **`foundation.test.ts:855`** hardcodes `"shared/components/surface.tsx"` in the
  allowlist exempting the KPI cell's severity and delta line from a type-role rule.
  It must name `metrics/KpiCell.tsx` or the gate silently stops covering it. Audit
  the whole allowlist per move rather than search-and-replacing.
- **Ten documentation files** reference these paths, 23 references in total:
  DESIGN.md, component-architecture.md, components.md, data-fetching.md,
  design-tokens.md, layout-stability.md, performance.md, SKILL.md, testing.md,
  typescript-and-react.md, plus web/AGENTS.md. They move in the same change, per
  AGENTS.md on a fact told in two layers.
- **The `FULL_BLEED` / `BLEED_INSET` rows were fixed upstream, not here.** This
  spec found them advertised as exports of `@/shared/components/surface` with
  neither identifier anywhere in `web/src`, so an agent following the table wrote
  an import that did not compile. The redesign removed the rows and rewrote
  layout.md's paragraph to name the two halves a band writes out by hand before
  this change landed, so nothing here touches either file for that reason.

## How it was done

The mechanical parts were scripted, because 200-odd files of import rewriting is
not hand work and neither is 34 computed import headers.

1. Directories created, the 15 loose files moved whole (`git mv`), specifiers
   rewritten tree-wide.
2. `surface.tsx` and `ui.tsx` split by **line range**, so every comment moved
   byte-for-byte rather than being retyped. The import header for each new file
   was **computed** from which identifiers the block actually references, which is
   what kept 34 headers free of both an unused import and a missing one.
3. Call sites rewritten by a codemod, not by `sed`: one grouped statement
   (`import { Dot, Section, RowAction } from ".../surface"`) becomes three
   statements pointing at three directories, and the specifier list spans lines.
   It refuses a statement it cannot fully map rather than dropping a name, which
   is how the one import carrying a comment inside its braces was caught instead
   of silently losing `StatCard`.
4. `hooks.ts` split the same way. It was already ordered by domain, so the domains
   were contiguous runs and the long section banners (which carry the reasoning
   for the tenant-scoped/deployment-wide key split) travelled with their sections.

Three traps worth recording, all of which cost a cycle:

- **`sed`'s longest-suffix trap.** A first pass looped over a `"Name:dir"` string
  in zsh, which does not word-split an unquoted variable, so the loop ran once and
  `${pair##*:}` resolved to the *last* colon in the whole string. `DataTable`
  landed at `components/access/DataTable`. It typechecked as a missing module, so
  it surfaced, but a mapping that had happened to be plausible would not have.
- **Anchor a specifier rewrite on its closing quote.** `components/Field"` and
  `components/FieldMessages"` differ only by suffix.
- **A block boundary can be off by one declaration and still compile.**
  `optionKey` / `optionValue` sat between `Badge` and `FilterSelect` and went with
  the wrong one; the symptom was an unused-variable error in one file and an
  undefined name in the other, not anything about boundaries.

## Verification

`lint`, `typecheck`, 135 test files / 3,844 tests, and a production build, all
green. Two notes on reading those runs:

- The test count **rises** rather than staying flat, which is expected and not a
  sign that cases were added: several gates are `it.each(sources)` over the source
  tree, so 34 more files means more generated cases.
- A full-suite run mid-refactor reported five failures in `ActivityPage`,
  `UsagePage` and `UsagePageScope` that passed in isolation and passed on the next
  full run. That is **pre-existing load-dependent flakiness**, not the refactor:
  the names include "frames the active preset when the two initial windows are read
  a millisecond apart". Worth a separate look; `pytest.ini`'s no-blanket-rerun
  reasoning in the root AGENTS.md applies to Vitest for the same reason.

The behavioral e2e suite has not been run against this; it needs a built gateway
and is the remaining check.

## Follow-ups this deliberately left

- **Delete `deprecated/SettingsSection`.** No call site anywhere. Removing a
  component is a maintainer's call, not a side effect of moving files, so it is
  pinned at zero by a test instead.
- **Convert the remaining deprecated call sites.** `PageHeader` on 4 pages,
  `StatCard` on 4 uses in `OverviewPage`, `RowActions` on 1 in `PasskeysCard`. A
  taxonomy change that also rewrote six pages would stop being reviewable by
  shape. One page at a time, which is what DESIGN.md already invites.
- **Split `tabs.test.tsx`** if its header docstring is ever separable per
  component.
- **Move the remaining poll cadences out of `queryKeys.ts`.** `BUILD_POLL_MS`,
  `HEALTH_POLL_MS`, `MAINTENANCE_MODE_POLL_MS`, `PROVIDER_HEALTH_REFRESH_MS` and
  `NO_RETRY` are query *options*, not keys, and each is read by exactly one domain
  module. They stayed because the constant block moved whole, with its comments,
  and cutting it into 43 individually-bounded pieces was the one place where the
  risk was not worth the tidiness. `PROVIDER_HEALTH_REFRESH_MS` already moved to
  `providers.ts`, because it was the one constant a page reached for and so the one
  blocking the boundary rule below. The keys themselves must not be split: 109
  invalidations reach 39 of them across domains.
- **`web/design/actions.md` still opens on `Button`.** The file is now named for
  the whole action family; its first section is about the three variants only.
