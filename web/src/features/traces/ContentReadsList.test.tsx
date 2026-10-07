import { render, screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import type { ContentAccess } from "@/client"
import {
  ContentReadsList,
  readerLabel,
} from "@/features/traces/ContentReadsList"

function read(overrides: Partial<ContentAccess> = {}): ContentAccess {
  return {
    accessed_at: "2026-10-07T12:00:00Z",
    trace_id: "s-0123456789abcdef",
    span_id: "req-1",
    reader_kind: "owner",
    reader: "user:11111111-aaaa",
    reason: null,
    ...overrides,
  }
}

describe("ContentReadsList", () => {
  it("names each read's kind and a break-glass read's reason", () => {
    render(
      <ContentReadsList
        reads={[
          read(),
          read({
            span_id: "req-2",
            reader_kind: "admin",
            reader: "user:22222222-bbbb",
          }),
          read({
            span_id: "req-3",
            reader_kind: "break_glass",
            reader: "master_key",
            reason: "Court order received from counsel",
          }),
        ]}
        names={new Map([["11111111-aaaa", "Alice Liddell"]])}
        isLoading={false}
      />,
    )

    const rows = screen.getAllByRole("row").slice(1)
    expect(rows).toHaveLength(3)
    expect(within(rows[0]).getByText("Owner")).toBeInTheDocument()
    expect(within(rows[0]).getByText("Alice Liddell")).toBeInTheDocument()
    expect(within(rows[1]).getByText("Organization admin")).toBeInTheDocument()
    expect(within(rows[1]).getByText("Identity 22222222")).toBeInTheDocument()
    expect(within(rows[2]).getByText("Break-glass")).toBeInTheDocument()
    expect(within(rows[2]).getByText("Master key")).toBeInTheDocument()
    expect(
      within(rows[2]).getByText("Court order received from counsel"),
    ).toBeInTheDocument()
    expect(within(rows[2]).getByText("s-01234567")).toBeInTheDocument()
  })

  it("says when no one has read the workspace's content", () => {
    render(<ContentReadsList reads={[]} names={new Map()} isLoading={false} />)

    expect(
      screen.getByText("No one has read this workspace's content yet."),
    ).toBeInTheDocument()
  })
})

describe("readerLabel", () => {
  it("keeps a reader it does not recognize as it is", () => {
    expect(readerLabel("service:x", new Map())).toBe("service:x")
  })
})
