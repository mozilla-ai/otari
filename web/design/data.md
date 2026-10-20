# Data display

## Which shape?

```text
Is it a list of records with more than one attribute each?
 └── DataTable, inside a TableScrollFrame if it is wide
Anything else on this list is a metric, and lives in [metrics.md](metrics.md):
 └── headline numbers -> KpiStrip · a change -> TrendChip · a limit -> SpendMeter
     a quantity over time -> TrendChart or Sparkline · a part-to-whole -> a ramp
```

## DataTable

```ts
DataTable: { ariaLabel, columns, rows, getRowKey, isLoading?, emptyContent?,
  selectionMode?: "none" | "multiple", selectedKeys?, onSelectionChange?,
  disabledKeys?, sortDescriptor?, onSortChange?, onRowAction?, rowClassName?,
  isResizable?, detailKey?, renderDetail?, mobileLayout?: "table" | "cards",
  scrollResetKey? }

DataTableColumn: { id, header, cell: (row) => ReactNode, align?: "start" | "end",
  isRowHeader?, allowsSorting?, width?, minWidth? }
```

Note the names: `id` not `key`, `header` not `label`, `cell` not `renderCell`, and
`align: "end"` not `"right"`. `align` moves the header with the cell. One column
carries `isRowHeader` (usually the name); that is the cell a screen reader announces
as the row's identity.

The trailing action column takes `header: "Actions"`, not an empty string: the header
row is a set of labels and a blank one reads as a missing label, not as a deliberate
gap. `emptyContent` renders only once loading has finished, so it never flashes
"nothing matches" during a first fetch.

`mobileLayout="cards"` is for a table too wide to read at phone width. Under 768px it
renders one card per row instead of the grid: the `isRowHeader` column is the title, a
column with id `actions` sits beside it, and every other column is a label and value
line. Selection, `onRowAction` and `renderDetail` carry over; sorting and resize do not.
Without it the grid stays and scrolls sideways.

A white body on the page canvas, ruled by hairlines. The header is the accent tint
(`--color-primary-subtle`) set in the `text-overline` role, so it never outweighs
the rows it labels.

```tsx
// Correct
<TableScrollFrame className="otari-keys-table">
  <DataTable
    ariaLabel="API keys"
    columns={columns}
    rows={rows}
    getRowKey={(row) => row.id}
    selectionMode="multiple"
    selectedKeys={selected}
    onSelectionChange={setSelected}
    isLoading={query.isLoading}
    emptyContent={<EmptyMessage>No keys match this filter.</EmptyMessage>}
    detailKey={openRow}
    renderDetail={(row) => <KeyDetail row={row} />}
  />
</TableScrollFrame>

// Incorrect: a hand-rolled table loses selection, keyboard grid navigation,
// sort and resize, and it will not pick up the .otari-table rules.
<table className="w-full">…</table>
```

Row states, and the only three:

| State | Ground |
| --- | --- |
| Rest | **No fill.** The page ground shows through, divided by a `border-subtle` hairline. The name is the only emphasis in the row |
| Hover | `surface-muted`. The whole row is the hit area, so the whole row lights |
| Selected | `primary-subtle`. Translucent, so it still reads over a hovered row |

Never zebra striping, never vertical rules between cells, never a bolder header.

## Building a table

Every table follows the same practice, and most of it arrives with `DataTable`
rather than being written at the call site.

**Past ten rows a table scrolls inside itself, under a pinned header.**
`DataTable` measures the header and the first `TABLE_VISIBLE_ROWS` rows as
rendered and marks its scroller; `design-system.css` turns the mark into a
`max-height`. Measured, not computed from a row height, because rows differ
from table to table: a two-line model row caps taller than a one-line key row,
at the same ten rows. The cap never exceeds `--table-max-height`, the viewport
less the top bar, a toolbar and a pager, so a table with very thick rows shows
fewer than ten and the pager stays in reach. A table of ten rows or fewer is its
own height. The header turns opaque only while capped, on the page's fill or,
inside a HeroUI `Card`, the card's; a table on any other surface sets
`--table-fill` on its section. A focused row scrolls clear of the header. The rows hand the wheel
back to the page at either end, and the cut is left where it falls, the same as
the horizontal clip. From `md` up, and never on paper: phones get their own
strategy.

Never write a `max-height` or an `overflow-y` on a table, its frame, or HeroUI's
scroller. The cap is one rule, and a second one beside it decides nothing.

**More rows than a page pages, in one of two ways:**

| The endpoint | Pages through | |
| --- | --- | --- |
| Pages itself (`skip`/`limit` and a count) | `TablePagination` driving the query | Activity, Sessions, Members |
| Returns every row at once | `useClientPagination(rows, resetKey)` into `TablePagination` | Usage breakdowns, health checks |

`useClientPagination`'s `resetKey` returns to the first page when it changes;
pass something that moves with the filters, never the rows, which a refetch
replaces. Either way, pass the page as `scrollResetKey` so a new page opens at
its first row. A hand-written `slice()` with a "Show all" button is the thing
this replaces: it grows the page by the whole remainder in one press.

A table that shows every row, because nothing pages it, is still bounded by the
ten-row cap.

**The controls stay outside the scroll.** Filters, tabs and view toggles sit
directly above the table and the pager directly below it, each a sibling of
the `DataTable` or of the `TableScrollFrame` around it. Nothing that configures
the table goes inside it, where it would scroll away with the rows.

**A row detail opens where the design puts it**, and the table does not decide:
a page of its own (Models, a drill-in to Activity), a modal or side panel
(Sessions), or inline under the row through `detailKey`/`renderDetail`
(Activity, Routing). An inline panel opening below the visible rows is scrolled
into view inside the table, never by moving the page, and it does not count
toward the ten rows.

## The base decides; a per-table class carries only its own lanes

`.otari-table` is the whole treatment: no fill on the root or the header, no
radius, no padding, no column separators, and the row separator on the faint
tier. **A table is a region of the one surface**, bounded by the section rules
around it and by its own hairlines.

That is worth stating because it was not true until recently and the shape of
the mistake is instructive. HeroUI's `Table.Root` ships as a card, the base rule
only half-neutralized it (it swapped HeroUI's fill for `--color-surface` and gave
the header a `--color-primary-subtle` tint), and so all sixteen per-table classes
carried the same three declarations to undo it:

```css
/* What each of sixteen classes said, each commented "Same terms as the others" */
.otari-keys-table .otari-table.table-root      { background-color: transparent; }
.otari-keys-table .otari-table .table__header  { background-color: transparent; … }
.otari-keys-table .otari-table .table__row, …  { border-color: var(--color-border-subtle); }
```

Forty-eight copies of one decision, because there was nowhere to say it once.
The base is now those terms and the copies are gone.

**So a per-table class is for what is genuinely that table's:** its row height,
its lane widths (keyed on the header's `[data-key="…"]`, so a lane is named rather
than counted), its sticky columns, and its own outer rules where it has them.
If you find yourself writing one of the three above, the base already says it and
`src/styles/foundation.test.ts` will reject the copy.

Two consequences worth knowing. Sixteen consumers overriding a default with no
exceptions is not a default: it is a base value that was wrong, and the fix is
to move it rather than to keep overriding. And the three cards that render a bare
`DataTable` with no wrapper class (`OrganizationBudgetsCard`,
`SpendCeilingsCard`, `OrganizationRosterCard`) were the ones still showing the
old treatment, so correcting the base is what converted them: their headers lose
a teal tint and their row separators move to the faint tier.
The lane does the aligning; see [layout.md](layout.md).

**`TableScrollFrame`'s `className` is a declared place, not a free label.** It
tracks horizontal scroll so a first column can pin, and the per-page class is what
`globals.css` hangs that page's column widths and pinning off. The list is closed:
`otari-keys-table`, `otari-models-table`, `otari-providers-table`,
`otari-routing-table`, `otari-domains-table`, `otari-members-table`,
`otari-provider-keys-table`, `otari-offered-models-table`,
`otari-workspaces-table`, `otari-accounts-table`, `otari-activity-table`,
`otari-budgets-table`, `otari-mcp-table`, `otari-offerings-table`,
`otari-sessions-table`. Inventing one
at the call site compiles and styles nothing.

A new page adds its class to `src/styles/globals.css`, beside the others, and the
block is small: the per-page rules there set column widths on
`.otari-<page>-table .table__column[data-key="…"]` and, where a column pins, its
sticky treatment on that header and on
`.otari-<page>-table .table__cell[data-column="…"]`. A body cell's `data-key` is a
generated react-aria key, not the column id, so a rule on it matches nothing: the
header pins and its cells scroll away underneath. Copy the nearest existing block;
do not invent properties. If
your table needs no column widths, you still need the class for the scroll tracking,
and an empty selector is not enough: add it to the shared group at the top of that
section instead.

Column rules: right-align a number, left-align everything else. A number in a cell
is `text-mono-caption` so digits line up between rows. An id is a `CopyableValue`,
which keeps the press off the row so a drag selects text instead of toggling
selection.

## TablePagination, BulkActionBar

```ts
TablePagination: { page, pageSize, total, rowsOnPage, onPageChange,
  onPageSizeChange, pageSizeOptions?, isFetching?, hasNextFallback? }
```

`page` is zero-based. `rowsOnPage` is separate from `total` on purpose: it is how
many rows this page actually rendered, which is what the range reads from, and
`total` may be null when the endpoint does not count. `hasNextFallback` is how a
non-counting endpoint says there is more.

`TablePagination` carries the range, the page size and two bare arrows. Page sizes
are `PAGE_SIZE_OPTIONS` (15, 30, 45, 60); do not invent a fifth, and read a size
from a URL through `snapPageSize`. A page past the end moves to the last page
that holds rows once the total is known. The controls hide while every row fits
one page at the smallest size and nothing promises more, so a short table has
no pager to explain; the range stays as an unseen live region, which is what a
screen reader hears a result count from. A table whose endpoint returns every
row pages it through `useClientPagination`. There is no rule
under it, because the page ends there.

Known cost, accepted: with the boxes gone, "which arrow can I press" rests on
`opacity: 0.4` alone at glyph size.

```ts
BulkActionBar: { selectedCount, allMatching, matchingTotal, canSelectAllMatching,
  onSelectAllMatching, onClear, children }
```

Every prop except `children` is required, including the four that describe the
select-all-matching affordance. Pass `allMatching: false` and
`canSelectAllMatching: false` when the page does not offer it.

`BulkActionBar` appears while a selection exists, **fixed near the bottom of the
viewport rather than in flow.** That is deliberate: an in-flow bar rendered above
the table on first selection and shifted every row down under the operator's
cursor mid-click-spree, and it scrolled out of view on long pages. Fixed gives zero
layout shift and stays reachable however deep the selection goes.

It carries the count, a Clear, the actions, and, once the visible page is fully
selected and more rows match the filter, a "select all N matching" affordance. The
destructive action names its count (`Revoke 3`), because a bulk action's blast
radius is the thing worth reading twice.
