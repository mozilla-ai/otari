import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { BreakGlassLog } from "@/features/traces/BreakGlassLog"
import { API_ROOT } from "@/shared/api/client"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("BreakGlassLog", () => {
  it("lists who opened which session, when and why", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          items: [
            {
              id: "1",
              kind: "organization",
              organization_id: "o",
              workspace_id: "w",
              trace_id: "s-one",
              session_user_id: "u",
              reader_user_id: null,
              reader_email: "owner@example.test",
              reason: "HR matter 17",
              reference: null,
              read_at: "2026-10-23T10:00:00Z",
            },
          ],
          has_more: false,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )
    render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <BreakGlassLog />
      </QueryClientProvider>,
    )

    expect(
      await screen.findByText(/owner@example.test opened session s-one/),
    ).toBeInTheDocument()
    expect(screen.getByText("“HR matter 17”")).toBeInTheDocument()
    expect(String(fetch.mock.calls[0][0])).toBe(
      `${API_ROOT}/organizations/me/traces/audit/break-glass?skip=0&limit=50`,
    )
  })

  it("pages to openings past the newest fifty", async () => {
    const read = (n: number) => ({
      id: String(n),
      kind: "organization",
      organization_id: "o",
      workspace_id: "w",
      trace_id: `s-${n}`,
      session_user_id: "u",
      reader_user_id: null,
      reader_email: "owner@example.test",
      reason: `reason ${n}`,
      reference: null,
      read_at: "2026-10-23T10:00:00Z",
    })
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (input) => {
        const older = String(input).includes("skip=50")
        const items = older
          ? [read(50)]
          : Array.from({ length: 50 }, (_, n) => read(n))
        return new Response(JSON.stringify({ items, has_more: !older }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      })
    render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <BreakGlassLog />
      </QueryClientProvider>,
    )

    await userEvent.click(
      await screen.findByRole("button", { name: "Show older openings" }),
    )

    expect(await screen.findByText("“reason 50”")).toBeInTheDocument()
    expect(
      fetch.mock.calls.some(([input]) => String(input).includes("skip=50")),
    ).toBe(true)
    expect(
      screen.queryByRole("button", { name: "Show older openings" }),
    ).not.toBeInTheDocument()
  })
})
