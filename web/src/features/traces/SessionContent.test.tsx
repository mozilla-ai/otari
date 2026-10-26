import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { TraceContentShare } from "@/client"
import { SessionContent } from "@/features/traces/SessionContent"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { traceDetail, traceSpan, traceSummary } from "@/tests/traceFixtures"

const SHARE: TraceContentShare = {
  id: "99999999-9999-9999-9999-999999999999",
  audience: "organization_admins",
  note: "flaky test",
  created_at: "2026-10-23T10:00:00Z",
  revoked_at: null,
  active: true,
  read_count: 1,
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

function mockApi(
  shares: TraceContentShare[] = [],
  more: TraceContentShare[] = [],
): Sent[] {
  const sent: Sent[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    sent.push({
      url: String(input),
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    const older = String(input).includes("skip=20")
    const body =
      method === "GET"
        ? older
          ? { items: more, has_more: false }
          : { items: shares, has_more: more.length > 0 }
        : SHARE
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
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>{ui}</DeploymentProvider>
    </QueryClientProvider>,
  )
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

  it("keeps an owner's share controls while the content cannot be opened", async () => {
    mockApi([SHARE])
    renderWith(
      <SessionContent
        detail={traceDetail({
          summary: traceSummary({ user_id: "alice" }),
          content_access: "owner",
          // A key-source outage: the content reads back empty.
          spans: [tool],
        })}
      />,
    )

    expect(
      await screen.findByRole("button", { name: "Stop sharing" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Share this session" }),
    ).not.toBeInTheDocument()
  })

  it("keeps ended shares listed with who read them", async () => {
    const reads = (email: string) => [
      {
        reader_user_id: null,
        reader_email: email,
        read_at: "2026-10-20T11:00:00Z",
      },
    ]
    mockApi([
      {
        ...SHARE,
        id: "ended",
        active: false,
        revoked_at: "2026-10-21T10:00:00Z",
        reads: reads("ended@example.test"),
      },
      {
        ...SHARE,
        id: "older",
        active: false,
        revoked_at: "2026-10-22T10:00:00Z",
        reads: reads("older@example.test"),
        read_count: 12,
      },
    ])
    renderWith(
      <SessionContent
        detail={traceDetail({
          summary: traceSummary({ user_id: "alice" }),
          content_access: "owner",
          spans: [{ ...tool, content: { command: "pytest -x" } }],
        })}
      />,
    )

    expect(
      await screen.findByText(/Read by ended@example.test/),
    ).toBeInTheDocument()
    expect(screen.getByText(/Read by older@example.test/)).toBeInTheDocument()
    expect(screen.getAllByText(/, ended /)).toHaveLength(2)
    expect(screen.getByText("The newest 1 of 12 reads.")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Stop sharing" }),
    ).not.toBeInTheDocument()
  })

  it("pages to older shares, so a running one past the first page can still be ended", async () => {
    const sent = mockApi(
      [SHARE],
      [{ ...SHARE, id: "older", note: "older one" }],
    )
    renderWith(
      <SessionContent
        detail={traceDetail({
          summary: traceSummary({ user_id: "alice" }),
          content_access: "owner",
          spans: [{ ...tool, content: { command: "pytest -x" } }],
        })}
      />,
    )

    await userEvent.click(
      await screen.findByRole("button", { name: "Show older shares" }),
    )

    expect(await screen.findByText("“older one”")).toBeInTheDocument()
    expect(sent.some((request) => request.url.includes("skip=20"))).toBe(true)
    expect(
      screen.getAllByRole("button", { name: "Stop sharing" }),
    ).toHaveLength(2)
  })

  it("lets an owner open a private session by break-glass, with a reason", async () => {
    const sent = mockApi()
    const opened = vi.fn()
    const detail = traceDetail({ spans: [{ ...tool, content_private: true }] })
    renderWith(<SessionContent detail={detail} onBreakGlass={opened} />)

    await userEvent.click(
      screen.getByRole("button", { name: "Open by break-glass" }),
    )
    const submit = screen.getByRole("button", { name: "Open and record" })
    expect(submit).toBeDisabled()
    await userEvent.type(
      screen.getByRole("textbox", { name: /Reason/ }),
      "HR matter 17, approved by legal",
    )
    await userEvent.click(submit)

    const post = sent.find((request) => request.method === "POST")
    expect(post?.url).toBe(
      `${API_ROOT}/organizations/me/traces/${detail.summary.trace_id}/break-glass?workspace_id=${detail.summary.workspace_id}`,
    )
    expect(post?.body).toEqual({ reason: "HR matter 17, approved by legal" })
    await vi.waitFor(() => expect(opened).toHaveBeenCalled())
  })

  it("shows a failed break-glass attempt's error only until the dialog closes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) =>
      (init?.method ?? "GET").toUpperCase() === "POST"
        ? new Response(JSON.stringify({ detail: "The reason is too short" }), {
            status: 422,
            headers: { "Content-Type": "application/json" },
          })
        : new Response(JSON.stringify({ items: [], has_more: false }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
    )
    const detail = traceDetail({ spans: [{ ...tool, content_private: true }] })
    renderWith(<SessionContent detail={detail} onBreakGlass={vi.fn()} />)
    const open = () =>
      userEvent.click(
        screen.getByRole("button", { name: "Open by break-glass" }),
      )

    await open()
    await userEvent.type(
      screen.getByRole("textbox", { name: /Reason/ }),
      "HR matter 17, approved by legal",
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Open and record" }),
    )
    expect(
      await screen.findAllByText(/The reason is too short/),
    ).not.toHaveLength(0)

    await userEvent.click(screen.getByRole("button", { name: "Cancel" }))
    await open()

    expect(
      screen.queryByText(/The reason is too short/),
    ).not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: /Reason/ })).toHaveValue(
      "HR matter 17, approved by legal",
    )
  })

  it("offers no break-glass to someone who is not an owner", () => {
    mockApi()
    renderWith(
      <SessionContent
        detail={traceDetail({ spans: [{ ...tool, content_private: true }] })}
      />,
    )

    expect(
      screen.queryByRole("button", { name: "Open by break-glass" }),
    ).not.toBeInTheDocument()
  })

  it("says a break-glass opening is recorded and seen by every owner", () => {
    mockApi()
    renderWith(
      <SessionContent
        detail={traceDetail({
          content_access: "break_glass",
          spans: [{ ...tool, content: { command: "pytest -x" } }],
        })}
      />,
    )

    expect(
      screen.getByText(/every owner of the organization can see it/),
    ).toBeInTheDocument()
  })
})
