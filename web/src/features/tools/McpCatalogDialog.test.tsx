import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  McpCatalogServer,
  McpCatalogSyncStatus,
  WorkspaceMcpServer,
  WorkspaceMcpServerHealth,
} from "@/client"
import { McpCatalogDialog } from "@/features/tools/McpCatalogDialog"
import {
  mcpCatalogServer,
  mcpCatalogSyncStatus,
  workspaceMcpServer,
  workspaceMcpServerHealth,
} from "@/tests/fixtures"

const WORKSPACE = "11111111-1111-1111-1111-111111111111"

const SENTRY = mcpCatalogServer({
  icon_url: "https://mcp.sentry.example/icon.png",
  logo_url: "/api/v1/mcp-catalog/logos/io.example/sentry?v=abc",
})
const STRIPE = mcpCatalogServer({
  id: "99999999-9999-9999-9999-999999999999",
  name: "com.stripe/mcp",
  title: "Stripe",
  description: "Payments.",
  url: "https://mcp.stripe.example/",
  requires_token: true,
})
const LINEAR = mcpCatalogServer({
  id: "77777777-7777-7777-7777-777777777777",
  name: "app.linear/linear",
  title: "Linear",
  description: "Issues and projects.",
  url: "https://mcp.linear.example/mcp",
})

const LOADED = mcpCatalogSyncStatus({
  outcome: "complete",
  server_count: 3,
  is_stale: false,
  last_completed_at: "2026-11-12T08:00:00+00:00",
  finished_at: "2026-11-12T08:00:00+00:00",
})

interface Call {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

/**
 * Answers the catalog's reads, a sync request, the sign-in probe and the
 * create. `probes` maps a URL to what a check with no credential found;
 * `created` maps a server name to what its create answers: a stored server, or
 * a refusal's detail.
 */
function mockApi({
  status = LOADED,
  afterRequest = { ...LOADED, requested: true },
  popular = [LINEAR, SENTRY],
  servers = [SENTRY, STRIPE, LINEAR],
  probes = {},
  created = {},
}: {
  status?: McpCatalogSyncStatus
  afterRequest?: McpCatalogSyncStatus
  popular?: McpCatalogServer[]
  servers?: McpCatalogServer[]
  probes?: Record<string, WorkspaceMcpServerHealth>
  created?: Record<string, WorkspaceMcpServer | string>
} = {}): Call[] {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ url, method, body })
    if (url.endsWith("/mcp-catalog/sync")) {
      return Response.json(method === "POST" ? afterRequest : status, {
        status: method === "POST" ? 202 : 200,
      })
    }
    if (url.includes("/mcp-catalog/popular")) {
      return Response.json({ data: popular, count: popular.length })
    }
    if (url.includes("/mcp-catalog/servers")) {
      const params = new URL(url, "http://localhost").searchParams
      const query = (params.get("q") ?? "").toLowerCase()
      const matching = servers.filter((server) =>
        server.title.toLowerCase().includes(query),
      )
      const skip = Number(params.get("skip") ?? 0)
      const limit = Number(params.get("limit") ?? 25)
      return Response.json({
        data: matching.slice(skip, skip + limit),
        count: matching.length,
      })
    }
    if (url.endsWith("/mcp-servers/check")) {
      return Response.json(
        probes[String(body?.url)] ?? workspaceMcpServerHealth(),
      )
    }
    if (url.endsWith("/mcp-servers") && method === "POST") {
      const answer = created[String(body?.name)]
      if (typeof answer === "string") {
        return Response.json({ detail: answer }, { status: 409 })
      }
      return Response.json(
        answer ?? workspaceMcpServer({ name: String(body?.name) }),
        { status: 201 },
      )
    }
    return Response.json({})
  })
  return calls
}

function renderDialog(
  existing: WorkspaceMcpServer[] = [],
  onAddByUrl?: () => void,
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <McpCatalogDialog
        isOpen
        onOpenChange={() => {}}
        workspaceId={WORKSPACE}
        workspaceName="Alpha"
        existing={existing}
        onAddByUrl={onAddByUrl}
      />
    </QueryClientProvider>,
  )
}

function creates(calls: Call[]): Call[] {
  return calls.filter(
    (call) => call.method === "POST" && call.url.endsWith("/mcp-servers"),
  )
}

describe("McpCatalogDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("opens on the Popular servers, not on the whole catalog", async () => {
    const calls = mockApi()
    renderDialog()

    const popular = await screen.findByRole("list", { name: "Popular" })
    expect(within(popular).getByText("Linear")).toBeInTheDocument()
    expect(within(popular).getByText("Sentry")).toBeInTheDocument()
    expect(
      screen.getByText("Search to find any of the 3 servers in the catalog."),
    ).toBeInTheDocument()
    // Nothing reads the catalog's alphabetical first page before a search.
    expect(
      calls.some((call) => call.url.includes("/mcp-catalog/servers")),
    ).toBe(false)
    expect(
      screen.getByText("Listing a server here does not vouch for it."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add" })).toBeDisabled()
  })

  it("leaves out a server the workspace already has, on Popular and in a search", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    // The same endpoint, spelled with a trailing slash the catalog does not have.
    renderDialog([
      workspaceMcpServer({ url: "https://mcp.linear.example/mcp/" }),
    ])

    const popular = await screen.findByRole("list", { name: "Popular" })
    expect(within(popular).getByText("Sentry")).toBeInTheDocument()
    expect(screen.queryByText("Linear")).toBeNull()
    // The gateway is told the workspace, so its pages and counts leave them out too.
    expect(
      calls.some((call) =>
        call.url.endsWith(`/mcp-catalog/popular?workspace_id=${WORKSPACE}`),
      ),
    ).toBe(true)

    await user.type(
      screen.getByRole("searchbox", { name: "Search the catalog" }),
      "e",
    )
    const results = await screen.findByRole("list", { name: "Results" })
    expect(within(results).getByText("Stripe")).toBeInTheDocument()
    expect(screen.queryByText("Linear")).toBeNull()
    expect(
      calls.some(
        (call) =>
          call.url.includes("/mcp-catalog/servers") &&
          call.url.includes(`workspace_id=${WORKSPACE}`),
      ),
    ).toBe(true)
  })

  it("marks a maker's own server Official and says where it runs", async () => {
    mockApi({
      popular: [
        LINEAR,
        mcpCatalogServer({ title: "Sentry", publisher_kind: "community" }),
      ],
    })
    renderDialog()

    const popular = await screen.findByRole("list", { name: "Popular" })
    const [linear, sentry] = within(popular).getAllByRole("listitem")
    expect(within(linear).getByText("Official")).toBeInTheDocument()
    expect(
      within(linear).getByText(
        ". Runs on mcp.linear.example, its publisher's own domain",
      ),
    ).toBeInTheDocument()
    expect(within(sentry).queryByText("Official")).toBeNull()
  })

  it("keeps a pick in view under Selected once the search is cleared, listed once", async () => {
    mockApi()
    const user = userEvent.setup()
    renderDialog()
    await screen.findByRole("list", { name: "Popular" })
    const search = screen.getByRole("searchbox", { name: "Search the catalog" })

    await user.type(search, "stri")
    const results = await screen.findByRole("list", { name: "Results" })
    await user.click(
      within(results).getByRole("checkbox", { name: "Select Stripe" }),
    )
    await user.clear(search)

    const selected = await screen.findByRole("list", { name: "Selected" })
    expect(within(selected).getByText("Stripe")).toBeInTheDocument()
    expect(screen.getByText("1 selected")).toBeInTheDocument()
    await screen.findByRole("list", { name: "Popular" })

    // A Popular pick moves up too, so it is never listed twice.
    await user.click(screen.getByRole("checkbox", { name: "Select Linear" }))
    expect(screen.getAllByText("Linear")).toHaveLength(1)
    expect(
      within(screen.getByRole("list", { name: "Selected" })).getByText(
        "Linear",
      ),
    ).toBeInTheDocument()

    // Unticking it under Selected puts it back.
    await user.click(screen.getByRole("checkbox", { name: "Select Stripe" }))
    await user.click(screen.getByRole("checkbox", { name: "Select Linear" }))
    expect(screen.queryByRole("list", { name: "Selected" })).toBeNull()
    expect(
      within(screen.getByRole("list", { name: "Popular" })).getByText("Linear"),
    ).toBeInTheDocument()
  })

  it("offers the add-by-URL form for a server the catalog does not list", async () => {
    mockApi()
    const user = userEvent.setup()
    const onAddByUrl = vi.fn()
    renderDialog([], onAddByUrl)
    await screen.findByRole("list", { name: "Popular" })
    expect(screen.getByText("Not listed?")).toBeInTheDocument()

    await user.type(
      screen.getByRole("searchbox", { name: "Search the catalog" }),
      "granola",
    )
    expect(
      await screen.findByText('No server matches "granola".'),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Add it by URL" }))

    expect(onAddByUrl).toHaveBeenCalledOnce()
  })

  it("offers to load an empty catalog, for this workspace", async () => {
    const calls = mockApi({
      status: mcpCatalogSyncStatus(),
      afterRequest: mcpCatalogSyncStatus({ requested: true }),
      popular: [],
    })
    const user = userEvent.setup()
    renderDialog()

    expect(
      await screen.findByText(/The catalog has not been loaded yet/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("list", { name: "Popular" })).toBeNull()
    await user.click(screen.getByRole("button", { name: "Load catalog" }))

    expect(
      await screen.findByText(/Loading the catalog. This takes a couple/),
    ).toBeInTheDocument()
    const asked = calls.find(
      (call) =>
        call.method === "POST" && call.url.endsWith("/mcp-catalog/sync"),
    )
    expect(asked?.body).toEqual({ workspace_id: WORKSPACE })
  })

  it("says why the last load failed", async () => {
    mockApi({
      status: mcpCatalogSyncStatus({
        outcome: "failed",
        error: "the registry answered HTTP 503",
      }),
      popular: [],
    })
    renderDialog()

    expect(
      await screen.findByText(
        "The last attempt failed: the registry answered HTTP 503.",
      ),
    ).toBeInTheDocument()
  })

  it("says plainly when the catalog is turned off", async () => {
    mockApi({ status: mcpCatalogSyncStatus({ mode: "off" }), popular: [] })
    renderDialog()

    expect(
      await screen.findByText(/The catalog is turned off on this deployment/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Load catalog" })).toBeNull()
  })

  it("offers a refresh once the catalog is stale, and says when one is held back", async () => {
    mockApi({
      status: {
        ...LOADED,
        is_stale: true,
        outcome: "partial",
        next_request_at: "2026-11-12T09:05:00+00:00",
      },
    })
    renderDialog()

    const refresh = await screen.findByRole("button", { name: "Refresh" })
    expect(refresh).toBeDisabled()
    expect(screen.getByText(/It can be loaded again after/)).toBeInTheDocument()
    // The catalog it holds stays browsable meanwhile.
    expect(
      await screen.findByRole("list", { name: "Popular" }),
    ).toBeInTheDocument()
  })

  it("says nothing about loading while the catalog is fresh", async () => {
    mockApi()
    renderDialog()

    await screen.findByRole("list", { name: "Popular" })
    expect(screen.queryByRole("button", { name: "Refresh" })).toBeNull()
  })

  it("asks the gateway to search, and hides Popular while it does", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderDialog()
    await screen.findByRole("list", { name: "Popular" })

    await user.type(
      screen.getByRole("searchbox", { name: "Search the catalog" }),
      "stri",
    )

    const results = await screen.findByRole("list", { name: "Results" })
    expect(within(results).getByText("Stripe")).toBeInTheDocument()
    expect(within(results).queryByText("Sentry")).toBeNull()
    expect(screen.queryByRole("list", { name: "Popular" })).toBeNull()
    expect(calls.some((call) => call.url.includes("q=stri"))).toBe(true)
  })

  it("adds each pick through the create, in the mode chosen for it, and reports what came of each", async () => {
    const calls = mockApi({
      probes: {
        // Refuses a caller with no account, so it starts on Personal.
        [LINEAR.url]: workspaceMcpServerHealth({
          status: "failing",
          error: "unauthorized",
          http_status: 401,
        }),
      },
      created: {
        Linear: workspaceMcpServer({
          name: "Linear",
          credential_mode: "per_user",
          health: null,
        }),
        Stripe: workspaceMcpServer({
          name: "Stripe",
          health: workspaceMcpServerHealth({
            tools: ["a", "b"],
            tool_count: 2,
          }),
        }),
        "Sentry 2": "This workspace already holds 50 MCP servers",
      },
    })
    const user = userEvent.setup()
    // The workspace already has a server called Sentry, at another address.
    renderDialog([workspaceMcpServer({ name: "Sentry" })])

    await user.click(
      await screen.findByRole("checkbox", { name: "Select Linear" }),
    )
    await user.click(screen.getByRole("checkbox", { name: "Select Sentry" }))
    await user.type(
      screen.getByRole("searchbox", { name: "Search the catalog" }),
      "stri",
    )
    await user.click(
      await screen.findByRole("checkbox", { name: "Select Stripe" }),
    )
    expect(screen.getByText("3 selected")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Add 3 servers" }))
    expect(
      screen.getByRole("heading", { name: "Add 3 servers to Alpha" }),
    ).toBeInTheDocument()

    // A title already in use is numbered, so the add is not refused for it.
    const names = screen.getAllByRole("textbox", { name: /^Name/ })
    expect(names.map((field) => (field as HTMLInputElement).value)).toEqual([
      "Linear",
      "Sentry 2",
      "Stripe",
    ])
    const linear = screen.getByRole("region", { name: "Linear" })
    await waitFor(() =>
      expect(
        within(linear).getByRole("radio", { name: "Personal accounts" }),
      ).toBeChecked(),
    )
    // A Personal server takes no shared token.
    expect(
      within(linear).queryByLabelText(/Authorization token/),
    ).not.toBeInTheDocument()
    const stripe = screen.getByRole("region", { name: "Stripe" })
    expect(
      within(stripe).getByRole("radio", { name: "Shared account" }),
    ).toBeChecked()
    await user.type(
      within(stripe).getByLabelText("Authorization token for Stripe"),
      " secret-token ",
    )
    // Availability starts on On request, as a server added by URL does.
    expect(
      within(stripe).getByRole("radio", { name: /^On request/ }),
    ).toBeChecked()
    await user.click(
      within(stripe).getByRole("radio", { name: /^Every request/ }),
    )
    await user.click(screen.getByRole("button", { name: "Confirm" }))

    const outcomes = await screen.findByRole("list", { name: "What was added" })
    expect(
      screen.getByRole("heading", { name: "2 of 3 added to Alpha" }),
    ).toBeInTheDocument()
    const rows = within(outcomes).getAllByRole("listitem")
    expect(rows[0]).toHaveTextContent("Linear")
    expect(rows[0]).toHaveTextContent("their own account")
    expect(rows[1]).toHaveTextContent("NOT ADDED")
    expect(rows[1]).toHaveTextContent(
      "This workspace already holds 50 MCP servers",
    )
    expect(rows[2]).toHaveTextContent("CONNECTED")

    const sent = creates(calls)
    expect(
      sent.every((call) =>
        call.url.endsWith(`/workspaces/${WORKSPACE}/mcp-servers`),
      ),
    ).toBe(true)
    expect(sent.map((call) => call.body)).toEqual([
      {
        name: "Linear",
        url: LINEAR.url,
        enabled: true,
        credential_mode: "per_user",
        attach_by_default: false,
      },
      {
        name: "Sentry 2",
        url: SENTRY.url,
        enabled: true,
        credential_mode: "shared",
        attach_by_default: false,
      },
      {
        name: "Stripe",
        url: STRIPE.url,
        authorization_token: "secret-token",
        enabled: true,
        credential_mode: "shared",
        attach_by_default: true,
      },
    ])
    // The probes asked each server with no credential at all.
    const probed = calls.filter((call) => call.url.endsWith("/check"))
    expect(probed.every((call) => call.body?.authorization_token === "")).toBe(
      true,
    )
  })

  it("keeps the mode an admin picks over the suggestion", async () => {
    const calls = mockApi({
      probes: {
        [LINEAR.url]: workspaceMcpServerHealth({
          status: "failing",
          error: "unauthorized",
        }),
      },
    })
    const user = userEvent.setup()
    renderDialog()
    await user.click(
      await screen.findByRole("checkbox", { name: "Select Linear" }),
    )
    await user.click(screen.getByRole("button", { name: "Add" }))
    const linear = screen.getByRole("region", { name: "Linear" })
    await waitFor(() =>
      expect(
        within(linear).getByRole("radio", { name: "Personal accounts" }),
      ).toBeChecked(),
    )

    await user.click(
      within(linear).getByRole("radio", { name: "Shared account" }),
    )
    await user.type(
      within(linear).getByLabelText("Authorization token for Linear"),
      "lin_api_key",
    )
    await user.click(screen.getByRole("button", { name: "Confirm" }))
    await screen.findByRole("list", { name: "What was added" })

    expect(creates(calls)[0].body).toMatchObject({
      credential_mode: "shared",
      authorization_token: "lin_api_key",
    })
  })

  it("will not add two servers under one name", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderDialog()
    await user.click(
      await screen.findByRole("checkbox", { name: "Select Linear" }),
    )
    await user.click(screen.getByRole("checkbox", { name: "Select Sentry" }))
    await user.click(screen.getByRole("button", { name: "Add 2 servers" }))

    const [first] = screen.getAllByRole("textbox", { name: /^Name/ })
    await user.clear(first)
    await user.type(first, "Sentry")

    expect(
      screen.getAllByText("Two servers in this list have this name."),
    ).toHaveLength(2)
    expect(screen.getByRole("button", { name: "Confirm" })).toBeDisabled()

    await user.click(screen.getByRole("button", { name: "Back" }))
    expect(
      screen.getByRole("checkbox", { name: "Select Linear" }),
    ).toBeChecked()
    expect(creates(calls)).toEqual([])
  })

  it("pages through search results on the gateway", async () => {
    const many = Array.from({ length: 30 }, (_, index) =>
      mcpCatalogServer({
        id: `id-${index}`,
        name: `io.example/s${index}`,
        title: `Server ${String(index).padStart(2, "0")}`,
        url: `https://s${index}.example/mcp`,
      }),
    )
    const calls = mockApi({ servers: many })
    const user = userEvent.setup()
    renderDialog()
    await screen.findByRole("list", { name: "Popular" })

    await user.type(
      screen.getByRole("searchbox", { name: "Search the catalog" }),
      "server",
    )
    await screen.findByText("Server 00")
    await user.click(
      screen.getByRole("button", { name: "Next page, catalog servers" }),
    )

    expect(await screen.findByText("Server 25")).toBeInTheDocument()
    expect(calls.some((call) => call.url.includes("skip=25&limit=25"))).toBe(
      true,
    )
  })
})
