import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"
import type { Selection, SortDescriptor } from "react-aria-components"
import { CopyableValue } from "../actions/CopyField"
import { EmptyState } from "../feedback/EmptyState"
import type { DataTableColumn, DataTableProps } from "./DataTable"
import { DataTable } from "./DataTable"

interface KeyRow {
  id: string
  name: string
  owner: string
  requests: number
  status: "active" | "revoked"
}

const ROWS: KeyRow[] = [
  {
    id: "key_01JQZ8X2M4",
    name: "ci-pipeline",
    owner: "ops@example.com",
    requests: 18402,
    status: "active",
  },
  {
    id: "key_01JQZ8X2M5",
    name: "staging-app",
    owner: "dev@example.com",
    requests: 4211,
    status: "active",
  },
  {
    id: "key_01JQZ8X2M6",
    name: "notebook",
    owner: "data@example.com",
    requests: 87,
    status: "active",
  },
  {
    id: "key_01JQZ8X2M7",
    name: "old-demo",
    owner: "dev@example.com",
    requests: 0,
    status: "revoked",
  },
]

const COLUMNS: DataTableColumn<KeyRow>[] = [
  {
    id: "name",
    header: "Name",
    isRowHeader: true,
    allowsSorting: true,
    cell: (row) => row.name,
  },
  {
    id: "id",
    header: "Key id",
    cell: (row) => (
      <CopyableValue
        value={row.id}
        label="key id"
        className="font-mono text-caption"
      />
    ),
  },
  { id: "owner", header: "Owner", cell: (row) => row.owner },
  {
    id: "requests",
    header: "Requests",
    align: "end",
    allowsSorting: true,
    cell: (row) => row.requests.toLocaleString("en-US"),
  },
  {
    id: "status",
    header: "Status",
    cell: (row) => (
      <span className={row.status === "active" ? "text-success" : "text-muted"}>
        {row.status === "active" ? "Active" : "Revoked"}
      </span>
    ),
  },
]

// Annotated with the props type rather than `satisfies Meta<typeof DataTable>`:
// the component is generic in its row, and inferring it through `typeof` widens
// `Row` back to `object`, which then rejects every column and callback below.
const meta: Meta<DataTableProps<KeyRow>> = {
  title: "Design system/Data/DataTable",
  component: DataTable,
  args: {
    ariaLabel: "API keys",
    columns: COLUMNS,
    rows: ROWS,
    getRowKey: (row) => row.id,
  },
  parameters: { layout: "padded" },
}

export default meta

type Story = StoryObj<DataTableProps<KeyRow>>

/**
 * Columns and rows declared, everything else from react-aria: keyboard grid
 * navigation, sort, selection, resize. A page opts into the behaviors it needs
 * rather than hand-rolling a `<table>`.
 */
export const Default: Story = {}

/** A spinner in place of the body while the first page loads. */
export const Loading: Story = {
  args: { rows: [], isLoading: true },
}

/**
 * `emptyContent` takes whatever the page wants to say, which is usually an
 * `EmptyState` rather than a bare sentence.
 */
export const Empty: Story = {
  args: {
    rows: [],
    emptyContent: (
      <EmptyState
        title="No keys yet"
        description="A key authenticates a caller against this gateway."
      />
    ),
  },
}

/**
 * Multiple selection. The header box is the indeterminate case, and the revoked
 * row is in `disabledKeys`: `disabledBehavior` is fixed to "selection", so that
 * row still opens its detail on click and only its checkbox is inert.
 */
export const Selectable: Story = {
  render: (args) => {
    const [selected, setSelected] = useState<Selection>(
      new Set(["key_01JQZ8X2M5"]),
    )
    return (
      <DataTable
        {...args}
        selectionMode="multiple"
        selectedKeys={selected}
        onSelectionChange={setSelected}
        disabledKeys={["key_01JQZ8X2M7"]}
      />
    )
  },
}

/** Sorting is controlled: the table reports a descriptor, the page sorts the rows. */
export const Sortable: Story = {
  render: (args) => {
    const [sort, setSort] = useState<SortDescriptor>({
      column: "requests",
      direction: "descending",
    })
    const sorted = [...ROWS].sort((a, b) => {
      const key = sort.column as keyof KeyRow
      const order = a[key] < b[key] ? -1 : a[key] > b[key] ? 1 : 0
      return sort.direction === "descending" ? -order : order
    })
    return (
      <DataTable
        {...args}
        rows={sorted}
        sortDescriptor={sort}
        onSortChange={setSort}
      />
    )
  },
}

/**
 * Inline detail, accordion style: the panel opens directly under the row that was
 * clicked instead of below the table. The detail row is deliberately outside
 * react-aria's collection, so expanding costs O(1) rather than re-processing the
 * page of rows, which is also why it never joins selection or keyboard
 * navigation. Click a row.
 */
export const WithRowDetail: Story = {
  render: (args) => {
    const [openKey, setOpenKey] = useState<string | null>("key_01JQZ8X2M4")
    return (
      <DataTable
        {...args}
        onRowAction={(key) =>
          setOpenKey((current) => (current === key ? null : key))
        }
        detailKey={openKey}
        renderDetail={(row) => (
          <div className="flex flex-col gap-1 px-4 py-3">
            <span className="text-overline">Scope</span>
            <p className="text-caption">
              {row.name} may call any model, owned by {row.owner}.
            </p>
          </div>
        )}
      />
    )
  },
}

/** Draggable column resize handles. */
export const Resizable: Story = {
  args: { resizable: true },
}

/** `rowClassName` tints a row from the row's own data. */
export const RowClassName: Story = {
  args: {
    rowClassName: (row) =>
      row.status === "revoked" ? "opacity-60" : undefined,
  },
}
