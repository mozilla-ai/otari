import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { PAGE_SIZE_OPTIONS, TablePagination } from "./TablePagination"

const meta = {
  title: "Design system/Data/TablePagination",
  component: TablePagination,
  args: {
    page: 1,
    pageSize: 25,
    total: 412,
    rowsOnPage: 25,
    onPageChange: () => {},
    onPageSizeChange: () => {},
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof TablePagination>

export default meta

type Story = StoryObj<typeof meta>

export const FirstPage: Story = {}

export const MiddlePage: Story = {
  args: { page: 5 },
}

/** The last page, where "next" has nothing to go to. */
export const LastPage: Story = {
  args: { page: 17, rowsOnPage: 12 },
}

/** One page of results: both arrows are inert and the range covers everything. */
export const SinglePage: Story = {
  args: { total: 12, rowsOnPage: 12 },
}

export const NoResults: Story = {
  args: { total: 0, rowsOnPage: 0 },
}

/**
 * An endpoint that does not report a total. The range reads open-ended, and
 * `hasNextFallback` is what decides whether "next" is offered, the page knows it
 * fetched a full window, so there is probably another one.
 */
export const UnknownTotal: Story = {
  args: { total: null, hasNextFallback: true },
}

export const Fetching: Story = {
  args: { page: 3, isFetching: true },
}

/** Wired up, so the page-size select and the arrows actually move. */
export const Interactive: Story = {
  render: (args) => {
    const [page, setPage] = useState(1)
    const [pageSize, setPageSize] = useState(25)
    const total = 412
    const rowsOnPage = Math.max(
      0,
      Math.min(pageSize, total - (page - 1) * pageSize),
    )
    return (
      <TablePagination
        {...args}
        page={page}
        pageSize={pageSize}
        total={total}
        rowsOnPage={rowsOnPage}
        onPageChange={setPage}
        onPageSizeChange={(size) => {
          setPageSize(size)
          setPage(1)
        }}
        pageSizeOptions={PAGE_SIZE_OPTIONS}
      />
    )
  },
}
