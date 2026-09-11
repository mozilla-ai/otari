# Layout

A page is a vertical stack of **bands**. Each band sets its own rules and its own
vertical padding. For authenticated dashboard pages, there is no page-level gap or card: two bands are
separated by the hairline one of them draws.

## The bleed rule

Rules run the full width of the scroll area; content stays in the centered column.
That needs two elements, always. `Section` is the pair.

```tsx
// Correct: the band's rules reach the viewport, its content stays in the column
<Section className="border-y border-border" contentClassName="grid grid-cols-5">
  {cells}
</Section>

// Incorrect: one element cannot be both full-width and centered. On a wide
// viewport every rule stops at the column edge and the page reads as cards again.
<section className="border-y border-border mx-auto max-w-[112.5rem]">{cells}</section>
```

`bleed={false}` for a band nested inside a column rather than sitting directly in
the scroll area. The escape is `100cqw` against `<main>`, so a nested band that
still bleeds does not stop at its column: measured inside a 360px grid cell it came
out 1464px wide and painted 552px past the right edge of the page.

A band that is not a `<section>` (a header row, a page-level notice) writes the two
halves out: `otari-bleed` on the outer element, and
`mx-auto w-full max-w-[112.5rem] px-4 md:px-6` on the element inside it. There are no
constants for these; `Section` is the one place the pair is named.

## The band components

| Component | Props | Use for |
| --- | --- | --- |
| `Section` | `className`, `contentClassName`, `bleed = true`, children | Any band of a page |
| `PageIntro` | `title`, `action?`, children | The opening of every page |
| `Toolbar` | `className?`, children | Above a table or list |
| `SettingsGroup` | `title?`, `description?`, `docsHref?`, `count?`, `bounded?`, children | A form page |
| `SettingRow` | `label`, `configKey?`, `help?`, `control`, `nested?`, `error?` | One setting inside a group |
| `KpiStrip` + `KpiCell` | see [metrics.md](metrics.md) | The metrics band |
| `TableScrollFrame` | `className`, children | Around a wide table |
| `ListDetail` + `ListDetailRow` | `listLabel`, `listAction?`, `list`, `empty?`, `onEmptyPress?`, `detail`, `detailLabel?`, `isDetailShown`, `onShowList`, `backLabel?` | A set of records where reading one is most of the work |

**`PageIntro` renders its own `<h1 className="text-display">`**, so never put a
heading inside it and never spell `text-display` on a page yourself. The sentence
under the title is `children`, not a `description` prop:

```tsx
// Correct
<PageIntro title="Keys" action={<Button variant="primary">Create key</Button>}>
  A key authenticates one application to the gateway.
</PageIntro>

// Incorrect: there is no `description` prop, and the h1 is already inside
<PageIntro description="A key authenticates one application.">
  <h1 className="text-display">Keys</h1>
</PageIntro>
```

**`Toolbar` already carries `.otari-toolbar`.** Do not pass the place class again;
`className` is for layout only.

`PageIntro` opens every page. The pre-redesign `PageHeader` it replaced is gone:
its last three call sites moved, and the component with them.

## Page recipes

A table page, in order:

```text
PageIntro          title, sentence, one primary action
KpiStrip           optional, ruled above and below
Toolbar            search, filters, refresh. FilterChips goes INSIDE it, and
                   renders the selects and their chips together
BulkActionBar      only while a selection exists, and fixed to the viewport
                   rather than in flow, so no row moves under the cursor
DataTable          the subject
TablePagination    range, page size, two bare arrows. No rule under it
```

A form page, in order:

```text
PageIntro
TabRow             only if the page has sibling surfaces
InfoBanner         only if there is a standing condition to state
SettingsGroup      one per topic, each with its own Save at its own foot
```

An autosaving settings page, which is the other shape:

```text
PageIntro          title, sentence, a trailing docsHref
SettingsGroup      bounded, one per topic
  SettingRow       label + key left, control right
  DisclosureRow    a list or an explanation that belongs to the row
```

No Save anywhere on that one: a text field commits on blur and on Enter, a
select on change. Reach for it when the settings are independent of one another,
so no single button could say what it is about to write.
`features/tools/ToolsGuardrailsPage` is the worked example.

A list-and-detail page, which is the shape for a set of records where reading
one is most of the work:

```text
PageIntro          title, sentence. No action: the create control goes in the
                   list column, because that is the column it lengthens
ErrorBanner        only if a read failed
ListDetail         the two columns, at 1:1.75, partitioned by a hairline
  ListDetailRow    one record: its name, and a line summarizing it
```

**No page reaches for this yet, and that is deliberate rather than an
oversight.** DESIGN.md's "What is deliberately not here" declines a primitive
with no consumer, and this is the standing exception: the frame was asked for as
a reusable shape rather than for one screen, so it ships with its stories and its
test and waits for the first page that suits it. Routing's policies were the
candidate and stayed a `DataTable`, because a flat set of policies has no
grouping or hierarchy for a detail column to earn. Reach for this when a record
has more to read than a row can hold.

The frame does not bleed: it is one framed object inside the page column, the
way a `bounded` settings group is, rather than rules running to the scroll
area's edges. Every fact a table would have put in a lane goes in the detail
column, since the list column holds a name and a line, and so do the controls
that act on the open record. **Nothing is edited in either column**: a record is
created and changed in a `FormDialog` over the page, which is the one create
surface (see [feedback.md](feedback.md)). **Below `md` one column shows at a
time**, and which one is a prop: two columns at 390px give neither a readable
measure, and stacking them puts every record above the one being read.

**A save that worked says nothing.** The control disables while the write is in
flight and that is the whole acknowledgement: a confirmation mark on every row
is one the reader learns to ignore by the third row, and it has to be held out
of flow to keep from moving the row it is congratulating. Only a refusal gets a
line.

`EmptyState`, `PageLoading` and a page-level `ErrorBanner` are bands like any other
and need no bleed helper; they already run the width they should.

A page component returns the bands as a fragment. There is no page wrapper and no
page-level gap: each band owns its vertical padding, and `<main>` supplies the
column.

## Rules for a settings group

**`SettingsGroup` draws its own row separators.** Its content is
`divide-y divide-border-subtle`, so **each direct child is one row** and the call
site never writes a border. Writing `border-border-subtle` on a child gives you two
lines.

```tsx
// Correct: three children, three rows, separators supplied
<SettingsGroup title="Access" description="Who can reach this gateway.">
  <SettingsRow label="Allow passkeys">…</SettingsRow>
  <SettingsRow label="Open signup">…</SettingsRow>
  <div className="flex justify-end"><Button variant="primary">Save access</Button></div>
</SettingsGroup>

// Incorrect: one child is one row, so this is a single row with a hand-drawn line
<SettingsGroup title="Access">
  <div className="flex flex-col divide-y divide-border-subtle">…</div>
</SettingsGroup>
```

**A row is a `SettingRow`.** Label and config key left, help under them, control
right in a lane every row shares; below `md` the control stacks full width. The
row draws no rule of its own, because the group divides its children. `nested`
indents it to `pl-8`, which is how a row says it belongs to the one above it: a
`DisclosureRow`'s panel is rows, not prose.

**The lane is a fixed-width slot and the control fills it**, by the repeated-row
rule further down this file. `w-full` on a field, `min-w-0 flex-1` on one sharing
the lane with a trailing button, `fullWidth` on a `FilterSelect`; a number keeps
`text-right tabular-nums` rather than a narrow box of its own. A control that
cannot fill a lane, a toggle or a copyable chip, sits at its leading edge. **No
row picks a control width**, for the reason it picks no field height: sized per
control the lane is not a lane, and a URL field, a select and a two-digit number
gave a column three different left edges.

`bounded` frames the rows inside the page column instead of bleeding them, and
the frame is also the dense place (`.otari-settings`), so its controls come out
32px on a desk and 36px at 16px on a phone. **No row picks a field height or a
field font size.** Use it where a page stacks several small groups, which run
together into one striped field as full-width bands. `docsHref` trails the
description with a `DocsLink`.

- Two border strengths and that is the whole hierarchy: `border-border-subtle`
  between rows (supplied), `border-border` around the group (supplied).
- **One Save per group, at that group's foot**, as the group's last child, on a
  group that submits as a unit. Never a floating Save for the whole page: a
  page-level Save cannot say what it is about to write. A group of independent
  settings autosaves instead and has no Save at all.
- `count` puts a muted number beside the title, for a group that lists things.

## Spacing

The step is 4px and everything is a multiple. The two values worth knowing by heart:
a page's column is padded `px-4 md:px-6` with `py-5 md:py-6`, and a KPI cell is
`px-7 py-[1.125rem]`.

Vary spacing deliberately: tighter to group, generous to separate. A uniform gap
everywhere reads as a wireframe.

## Repeated rows

Elements in repeated rows (a table, a rail, a list) must form vertical lanes. Use a
fixed-width slot with `flex-shrink-0` for icons, indicators and trailing actions,
**even when the slot is empty in some rows.** Never rely on `gap` alone to align a
column across rows whose content differs in length.

## Public authentication

Public auth and invitation pages use `features/auth/LoginPageShell`: one square,
bordered form card over the animated bar field selected for the sign-in redesign
([#991](https://github.com/mozilla-ai/otari/issues/991)). This is a deliberate
exception to the dashboard's flat bands, scoped to entry and account-recovery
flows. Keep the card horizontally centered and its top offset independent of
content height so errors and disclosures grow downward.

The decorative field uses the theme's primary color with changing opacity and
slightly rounded bars from the selected studio preset. These are illustration
marks, not rounded controls or tinted content surfaces. The corner marks and the
primary action keep their accent alongside the field; readable content stays on
an opaque semantic surface. Do not extend these exceptions to dashboard pages.

The Otari header and Mozilla AI footer are shared across these public pages,
including standalone deployments, as intentional product branding. Use existing
type roles and semantic tokens for both. The appearance control includes system,
light, and dark preferences; reduced motion renders a static field.
