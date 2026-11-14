import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { McpServerRequest, WorkspaceMcpServer } from "@/client"
import { McpRequestServerDialog } from "@/features/tools/McpRequestServerDialog"
import {
  mcpCatalogServer,
  mcpCatalogSyncStatus,
  workspaceMcpServer,
} from "@/tests/fixtures"

const WORKSPACE = "11111111-1111-1111-1111-111111111111"

const SENTRY = mcpCatalogServer({ publisher_kind: "vendor" })
const LINEAR = mcpCatalogServer({
  id: "77777777-7777-7777-7777-777777777777",
  name: "app.linear/linear",
  title: "Linear",
  description: "Issues and projects.",
  url: "https://mcp.linear.example/mcp",
  publisher_kind: "community",
})

function request(overrides: Partial<McpServerRequest> = {}): McpServerRequest {
  return {
    id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    workspace_id: WORKSPACE,
    catalog_name: LINEAR.name,
    title: LINEAR.title,
    url: LINEAR.url,
    note: null,
    status: "pending",
    decision_reason: null,
    decided_at: null,
    server_id: null,
    catalog_server: LINEAR,
    created_at: "2026-11-10T09:00:00+00:00",
    ...overrides,
  }
}

interface Call {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

/** Answers the catalog's reads and the member's own requests, which a send or a withdrawal then changes. */
function mockApi({
  mine = [],
  refusal,
}: {
  mine?: McpServerRequest[]
  /** Refuses a send with this detail. */
  refusal?: string
} = {}): Call[] {
  const calls: Call[] = []
  let own = [...mine]
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ url, method, body })
    if (url.endsWith("/mcp-catalog/sync")) {
      return Response.json(
        mcpCatalogSyncStatus({ outcome: "complete", server_count: 2 }),
      )
    }
    if (url.includes("/mcp-catalog/")) {
      return Response.json({ data: [SENTRY, LINEAR], count: 2 })
    }
    if (url.endsWith("/mcp-server-requests/mine")) {
      return Response.json({ data: own })
    }
    if (url.endsWith("/mcp-server-requests") && method === "POST") {
      if (refusal) return Response.json({ detail: refusal }, { status: 429 })
      const made = request({
        id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        catalog_name: String(body?.catalog_name),
        title: SENTRY.title,
        url: SENTRY.url,
        note: (body?.note as string | undefined) ?? null,
        catalog_server: SENTRY,
      })
      own = [made, ...own]
      return Response.json(made, { status: 201 })
    }
    if (url.endsWith("/acknowledge") && method === "POST") {
      own = own.filter((entry) => !url.includes(entry.id))
      return new Response(null, { status: 204 })
    }
    if (method === "DELETE") {
      own = own.filter((entry) => !url.endsWith(entry.id))
      return new Response(null, { status: 204 })
    }
    return Response.json({})
  })
  return calls
}

function renderDialog(existing: WorkspaceMcpServer[] = []) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <McpRequestServerDialog
        isOpen
        onOpenChange={() => {}}
        workspaceId={WORKSPACE}
        workspaceName="Alpha"
        existing={existing}
      />
    </QueryClientProvider>,
  )
}

describe("McpRequestServerDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists the catalog with a Request on each server, and nothing an admin would use", async () => {
    mockApi()
    renderDialog()

    const popular = await screen.findByRole("list", { name: "Popular" })
    expect(
      await within(popular).findByRole("button", { name: "Request Sentry" }),
    ).toBeVisible()
    expect(
      within(popular).getByRole("button", { name: "Request Linear" }),
    ).toBeVisible()
    expect(within(popular).getByText("Official")).toBeVisible()
    expect(screen.getByLabelText("Search the catalog")).toBeVisible()
    // No picking, no adding by URL, no sync: those are an admin's.
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add it by URL" }),
    ).not.toBeInTheDocument()
    expect(screen.queryByText("Your requests")).not.toBeInTheDocument()
  })

  it("sends a request with the note, then shows it as requested and waiting", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderDialog()

    await user.click(
      await screen.findByRole("button", { name: "Request Sentry" }),
    )
    expect(
      await screen.findByRole("heading", { name: "Request Sentry" }),
    ).toBeVisible()
    await user.type(
      screen.getByLabelText("What do you need it for?"),
      "  Triage production errors ",
    )
    await user.click(screen.getByRole("button", { name: "Send request" }))

    await waitFor(() =>
      expect(
        calls.find(
          (call) =>
            call.method === "POST" &&
            call.url.endsWith(`/workspaces/${WORKSPACE}/mcp-server-requests`),
        )?.body,
      ).toEqual({
        catalog_name: SENTRY.name,
        note: "Triage production errors",
      }),
    )
    // Back on the catalog: the row says Requested, and the request heads the dialog.
    const popular = await screen.findByRole("list", { name: "Popular" })
    expect(await within(popular).findByText("Requested")).toBeVisible()
    expect(
      within(popular).queryByRole("button", { name: "Request Sentry" }),
    ).not.toBeInTheDocument()
    const own = screen.getByRole("list", { name: "Your requests" })
    expect(within(own).getByText("Sentry")).toBeVisible()
    expect(within(own).getByText("Waiting for an admin")).toBeVisible()
  })

  it("sends no note when none was written", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderDialog()

    await user.click(
      await screen.findByRole("button", { name: "Request Linear" }),
    )
    await user.click(
      await screen.findByRole("button", { name: "Send request" }),
    )

    await waitFor(() =>
      expect(calls.find((call) => call.method === "POST")?.body).toEqual({
        catalog_name: LINEAR.name,
      }),
    )
  })

  it("says why a request was refused, and keeps the note", async () => {
    mockApi({
      refusal: "You asked for 10 servers in the last day; try again tomorrow",
    })
    const user = userEvent.setup()
    renderDialog()

    await user.click(
      await screen.findByRole("button", { name: "Request Sentry" }),
    )
    await user.type(
      await screen.findByLabelText("What do you need it for?"),
      "Errors",
    )
    await user.click(screen.getByRole("button", { name: "Send request" }))

    expect(
      await screen.findByText(
        "You asked for 10 servers in the last day; try again tomorrow",
      ),
    ).toBeVisible()
    expect(screen.getByLabelText("What do you need it for?")).toHaveValue(
      "Errors",
    )
  })

  it("shows the member's waiting and declined requests, withdraws a waiting one and dismisses a declined one", async () => {
    const calls = mockApi({
      mine: [
        request(),
        request({
          id: "cccccccc-cccc-cccc-cccc-cccccccccccc",
          catalog_name: "com.stripe/mcp",
          title: "Stripe",
          status: "declined",
          decision_reason: "Finance owns this",
          decided_at: "2026-11-11T09:00:00+00:00",
          catalog_server: null,
        }),
        // An approved request is the page's to announce, not this list's.
        request({
          id: "dddddddd-dddd-dddd-dddd-dddddddddddd",
          catalog_name: "io.sentry/mcp",
          title: "Sentry",
          status: "approved",
          server_id: "99999999-9999-9999-9999-999999999999",
        }),
      ],
    })
    const user = userEvent.setup()
    renderDialog()

    const own = await screen.findByRole("list", { name: "Your requests" })
    expect(within(own).getAllByRole("listitem")).toHaveLength(2)
    const [linear, stripe] = within(own).getAllByRole("listitem")
    expect(within(linear).getByText("Waiting for an admin")).toBeVisible()
    expect(within(stripe).getByText("Declined")).toBeVisible()
    expect(within(stripe).getByText(/Finance owns this/)).toBeVisible()
    // A declined request has nothing left to withdraw, and stays until dismissed.
    await user.click(
      within(stripe).getByRole("button", {
        name: "Dismiss your declined request for Stripe",
      }),
    )
    await waitFor(() =>
      expect(within(own).getAllByRole("listitem")).toHaveLength(1),
    )
    expect(
      calls.find((call) => call.url.endsWith("/acknowledge")),
    ).toMatchObject({
      method: "POST",
      url: expect.stringMatching(
        /mcp-server-requests\/cccccccc-cccc-cccc-cccc-cccccccccccc\/acknowledge$/,
      ),
    })

    await user.click(
      within(linear).getByRole("button", {
        name: "Withdraw your request for Linear",
      }),
    )

    await waitFor(() =>
      expect(calls.find((call) => call.method === "DELETE")?.url).toMatch(
        new RegExp(
          `/workspaces/${WORKSPACE}/mcp-server-requests/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa$`,
        ),
      ),
    )
    // Withdrawn, so it can be asked for again.
    expect(
      await screen.findByRole("button", { name: "Request Linear" }),
    ).toBeVisible()
  })

  it("leaves out a server the workspace already has", async () => {
    mockApi()
    renderDialog([workspaceMcpServer({ name: "Linear", url: LINEAR.url })])

    expect(
      await screen.findByRole("button", { name: "Request Sentry" }),
    ).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Request Linear" }),
    ).not.toBeInTheDocument()
  })
})
