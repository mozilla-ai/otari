import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { TraceContentShare } from "@/client"
import { SessionContent } from "@/features/traces/SessionContent"
import { API_ROOT } from "@/shared/api/client"
import { traceDetail, traceSpan, traceSummary } from "@/tests/traceFixtures"

const SHARE: TraceContentShare = {
  id: "99999999-9999-9999-9999-999999999999",
  audience: "organization_admins",
  note: "flaky test",
  created_at: "2026-10-23T10:00:00Z",
  expires_at: "2026-11-22T10:00:00Z",
  revoked_at: null,
  active: true,
  reads: [
    {
      reader_user_id: null,
      reader_email: "admin@example.test",
      read_at: "2026-10-23T11:00:00Z",
    },
  ],
}

interface Sent {
  url: string
  method: string
  body: unknown
}

function mockApi(shares: TraceContentShare[] = []): Sent[] {
  const sent: Sent[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    sent.push({
      url: String(input),
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    const body = method === "GET" ? { items: shares } : SHARE
    return new Response(JSON.stringify(body), {
      status: method === "POST" ? 201 : 200,
      headers: { "Content-Type": "application/json" },
    })
  })
  return sent
}

function renderWith(ui: ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

const tool = traceSpan({ span_id: "tool-1", kind: "tool" })

afterEach(() => {
  vi.restoreAllMocks()
})

describe("SessionContent", () => {
  it("tells someone else the content is private to the session's user", () => {
    mockApi()
    renderWith(
      <SessionContent
        detail={traceDetail({
          spans: [{ ...tool, content_private: true }],
        })}
      />,
    )

    expect(
      screen.getByText(/private to the person whose session it is/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Share this session" }),
    ).not.toBeInTheDocument()
  })

  it("tells a reader through a share whose say it was and that they were seen", () => {
    mockApi()
    renderWith(
      <SessionContent
        detail={traceDetail({
          content_access: "share",
          content_share: SHARE,
          spans: [{ ...tool, content: { command: "pytest -x" } }],
        })}
      />,
    )

    expect(screen.getByText(/Their note: “flaky test”/)).toBeInTheDocument()
    expect(
      screen.getByText(/They can see that you read it/),
    ).toBeInTheDocument()
  })

  it("lets the session's user share it and see who read it", async () => {
    const sent = mockApi([SHARE])
    const detail = traceDetail({
      summary: traceSummary({ user_id: "alice" }),
      content_access: "owner",
      spans: [{ ...tool, content: { command: "pytest -x" } }],
    })
    renderWith(<SessionContent detail={detail} />)

    expect(
      await screen.findByText(/Read by admin@example.test/),
    ).toBeInTheDocument()
    await userEvent.click(
      screen.getByRole("button", { name: "Share this session" }),
    )
    await userEvent.type(
      screen.getByRole("textbox", { name: "Why (optional)" }),
      "please look",
    )
    await userEvent.click(screen.getByRole("button", { name: "Share" }))

    const post = sent.find((request) => request.method === "POST")
    expect(post?.url).toBe(
      `${API_ROOT}/organizations/me/traces/${detail.summary.trace_id}/shares?workspace_id=${detail.summary.workspace_id}`,
    )
    expect(post?.body).toEqual({
      audience: "organization_admins",
      note: "please look",
      days: 30,
    })
  })

  it("offers no sharing on a session no user owns", () => {
    mockApi()
    renderWith(
      <SessionContent
        detail={traceDetail({
          summary: traceSummary({ user_id: null }),
          content_access: "owner",
          spans: [{ ...tool, content: { command: "pytest -x" } }],
        })}
      />,
    )

    expect(
      screen.queryByRole("button", { name: "Share this session" }),
    ).not.toBeInTheDocument()
  })
})
