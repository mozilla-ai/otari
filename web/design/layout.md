# Layout

A page is a vertical stack of **bands**. Each band sets its own rules and its own
vertical padding. There is no page-level gap and no card anywhere: two bands are
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
| `SettingsGroup` | `title?`, `description?`, `count?`, children | A form page |
| `KpiStrip` + `KpiCell` | see [metrics.md](metrics.md) | The metrics band |
| `TableScrollFrame` | `className`, children | Around a wide table |

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

`PageIntro`, not `PageHeader`. `PageHeader` is the pre-redesign card version still
live on 4 pages; see [DESIGN.md](DESIGN.md).

## Page recipes

A table page, in order:

```
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

```
PageIntro
TabRow             only if the page has sibling surfaces
InfoBanner         only if there is a standing condition to state
SettingsGroup      one per topic, each with its own Save at its own foot
```

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

There is no shared `SettingsRow` component; a row is a flex div the feature writes,
label left and control right in a shared lane. Copy the shape from a sibling group
rather than inventing a width.

- Two border strengths and that is the whole hierarchy: `border-border-subtle`
  between rows (supplied), `border-border` around the group (supplied).
- **One Save per group, at that group's foot**, as the group's last child. Never a
  floating Save for the whole page: a page-level Save cannot say what it is about to
  write.
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
