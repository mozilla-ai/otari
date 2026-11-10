import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  McpCatalogFeatured,
  McpCatalogServer,
  WorkspaceMcpServer,
} from "@/client"
import { McpCatalogDialog } from "@/features/tools/McpCatalogDialog"
import {
  mcpCatalogFeatured,
  mcpCatalogServer,
  organizationContext,
  workspaceMcpServer,
  workspaceMcpServerHealth,
} from "@/tests/fixtures"

const WORKSPACE = "11111111-1111-1111-1111-111111111111"

const SENTRY = mcpCatalogServer()
const STRIPE = mcpCatalogServer({
  id: "99999999-9999-9999-9999-999999999999",
  name: "com.stripe/mcp",
  title: "Stripe",
  description: "Payments.",
  url: "https://mcp.stripe.example/",
  requires_token: true,
})
const LINEAR = mcpCatalogFeatured()

interface Call {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

/**
 * Answers the catalog, the shelf and the create. `created` maps a server name
 * to what its create answers: a stored server, or a refusal's detail.
 */
function mockApi({
  servers = [SENTRY, STRIPE],
  featured = [LINEAR],
  created = {},
}: {
  servers?: McpCatalogServer[]
  featured?: McpCatalogFeatured[]
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
    if (url.endsWith("/organizations/me")) {
      return Response.json(organizationContext())
    }
    if (url.includes("/mcp-catalog/featured")) {
      return Response.json({ data: featured, count: featured.length })
    }
    if (url.includes("/mcp-catalog/servers")) {
      const query = new URL(url, "http://localhost").searchParams.get("q")
      const matching = query
        ? servers.filter((server) =>
            server.title.toLowerCase().includes(query.toLowerCase()),
          )
        : servers
      return Response.json({ data: matching, count: matching.length })
    }
    if (url.includes("/mcp-servers") && method === "POST") {
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

function renderDialog(existing: WorkspaceMcpServer[] = []) {
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
      />
    </QueryClientProvider>,
  )
}

function creates(calls: Call[]): Call[] {
  return calls.filter((call) => call.method === "POST")
}

describe("McpCatalogDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists what the organization recommends above the whole catalog", async () => {
    mockApi()
    renderDialog()

    const recommended = await screen.findByRole("list", {
      name: "Recommended by your organization",
    })
    expect(within(recommended).getByText("Linear")).toBeInTheDocument()
    const all = await screen.findByRole("list", { name: "All servers" })
    expect(within(all).getByText("Sentry")).toBeInTheDocument()
    expect(within(all).getByText("Stripe")).toBeInTheDocument()
    // Only the server whose registry entry asks for one is marked.
    expect(screen.getAllByText("Needs a token")).toHaveLength(1)
    expect(screen.getByRole("button", { name: "Review" })).toBeDisabled()
  })

  it("marks a server the workspace already has and does not offer it again", async () => {
    mockApi()
    // The same endpoint, spelled without the catalog's trailing slash.
    renderDialog([workspaceMcpServer({ url: "https://mcp.stripe.example" })])

    expect(await screen.findByText("Already added")).toBeInTheDocument()
    expect(
      screen.getByRole("checkbox", { name: "Select Stripe" }),
    ).toBeDisabled()
    expect(
      screen.getByRole("checkbox", { name: "Select Sentry" }),
    ).toBeEnabled()
  })

  it("says so when the catalog is empty, and still explains the other way in", async () => {
    mockApi({ servers: [], featured: [] })
    renderDialog()

    expect(await screen.findByText(/The catalog is empty/)).toBeInTheDocument()
    expect(screen.getByText(/add a server by its URL/)).toBeInTheDocument()
    expect(
      screen.queryByText("Recommended by your organization"),
    ).not.toBeInTheDocument()
  })

  it("asks the gateway to search, and hides the shelf while it does", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderDialog()
    await screen.findByText("Sentry")

    await user.type(
      screen.getByRole("searchbox", { name: "Search the catalog" }),
      "stri",
    )

    const results = await screen.findByRole("list", { name: "Results" })
    await waitFor(() =>
      expect(within(results).queryByText("Sentry")).not.toBeInTheDocument(),
    )
    expect(within(results).getByText("Stripe")).toBeInTheDocument()
    expect(
      screen.queryByText("Recommended by your organization"),
    ).not.toBeInTheDocument()
    expect(calls.some((call) => call.url.includes("q=stri"))).toBe(true)
  })

  it("adds each pick through the create and reports what came of each", async () => {
    const calls = mockApi({
      created: {
        Linear: workspaceMcpServer({
          name: "Linear",
          health: workspaceMcpServerHealth({
            tools: ["a", "b"],
            tool_count: 2,
          }),
        }),
        Stripe: workspaceMcpServer({
          name: "Stripe",
          health: workspaceMcpServerHealth({
            status: "failing",
            error: "unauthorized",
            http_status: 401,
          }),
        }),
        "Sentry 2": "This workspace already holds 50 MCP servers",
      },
    })
    const user = userEvent.setup()
    // The workspace already has a server called Sentry, at another address.
    renderDialog([workspaceMcpServer({ name: "Sentry" })])
    await screen.findByText("Stripe")

    await user.click(
      await screen.findByRole("checkbox", { name: "Select Linear" }),
    )
    await user.click(screen.getByRole("checkbox", { name: "Select Stripe" }))
    await user.click(screen.getByRole("checkbox", { name: "Select Sentry" }))
    expect(screen.getByText("3 selected")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Review" }))

    // A title already in use is numbered, so the add is not refused for it.
    const names = screen.getAllByRole("textbox", { name: /^Name/ })
    expect(names.map((field) => (field as HTMLInputElement).value)).toEqual([
      "Linear",
      "Stripe",
      "Sentry 2",
    ])
    await user.type(
      screen.getByLabelText("Authorization token for Stripe"),
      " secret-token ",
    )
    await user.click(screen.getByRole("button", { name: "Add 3 servers" }))

    const outcomes = await screen.findByRole("list", { name: "What was added" })
    expect(
      screen.getByRole("heading", { name: "2 of 3 added to Alpha" }),
    ).toBeInTheDocument()
    const rows = within(outcomes).getAllByRole("listitem")
    expect(rows[0]).toHaveTextContent("Linear")
    expect(rows[0]).toHaveTextContent("CONNECTED")
    expect(rows[1]).toHaveTextContent("NEEDS SIGN-IN")
    expect(rows[2]).toHaveTextContent("NOT ADDED")
    expect(rows[2]).toHaveTextContent(
      "This workspace already holds 50 MCP servers",
    )

    const sent = creates(calls)
    expect(sent.map((call) => call.body?.name)).toEqual([
      "Linear",
      "Stripe",
      "Sentry 2",
    ])
    expect(
      sent.every((call) =>
        call.url.endsWith(`/workspaces/${WORKSPACE}/mcp-servers`),
      ),
    ).toBe(true)
    expect(sent[0].body).toEqual({
      name: "Linear",
      url: "https://mcp.linear.example/mcp",
      enabled: true,
    })
    expect(sent[1].body?.authorization_token).toBe("secret-token")
    // A registry's description is never stored as a hint to the model.
    expect(sent.every((call) => !("purpose_hint" in (call.body ?? {})))).toBe(
      true,
    )
  })

  it("will not add two servers under one name", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderDialog()
    await screen.findByText("Stripe")
    await user.click(screen.getByRole("checkbox", { name: "Select Stripe" }))
    await user.click(screen.getByRole("checkbox", { name: "Select Sentry" }))
    await user.click(screen.getByRole("button", { name: "Review" }))

    const [first] = screen.getAllByRole("textbox", { name: /^Name/ })
    await user.clear(first)
    await user.type(first, "Sentry")

    expect(
      screen.getAllByText("Two servers in this list have this name."),
    ).toHaveLength(2)
    expect(screen.getByRole("button", { name: "Add 2 servers" })).toBeDisabled()

    await user.click(screen.getByRole("button", { name: "Back" }))
    expect(
      screen.getByRole("checkbox", { name: "Select Stripe" }),
    ).toBeChecked()
    expect(creates(calls)).toEqual([])
  })

  it("ticks only the row that was picked when the shelf and the catalog list different URLs", async () => {
    const calls = mockApi({
      servers: [SENTRY],
      featured: [
        mcpCatalogFeatured({
          name: SENTRY.name,
          title: "Sentry",
          url: "https://old.sentry.example/mcp",
          catalog_status: "changed",
        }),
      ],
    })
    const user = userEvent.setup()
    renderDialog()
    await screen.findByRole("list", { name: "All servers" })

    await waitFor(() =>
      expect(
        screen.getAllByRole("checkbox", { name: "Select Sentry" }),
      ).toHaveLength(2),
    )
    const [featured, listed] = screen.getAllByRole("checkbox", {
      name: "Select Sentry",
    })
    await user.click(featured)
    expect(featured).toBeChecked()
    expect(listed).not.toBeChecked()
    // Picking the catalog's row instead replaces it: one server, one URL.
    await user.click(listed)
    expect(featured).not.toBeChecked()
    expect(screen.getByText("1 selected")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Review" }))
    await user.click(screen.getByRole("button", { name: "Add 1 server" }))
    await screen.findByRole("list", { name: "What was added" })
    expect(creates(calls).map((call) => call.body?.url)).toEqual([SENTRY.url])
  })

  it("pages through the catalog on the gateway", async () => {
    const many = Array.from({ length: 25 }, (_, index) =>
      mcpCatalogServer({
        id: `id-${index}`,
        name: `io.example/s${index}`,
        title: `Server ${index}`,
        url: `https://s${index}.example/mcp`,
      }),
    )
    const calls: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      calls.push(url)
      if (url.endsWith("/organizations/me")) {
        return Response.json(organizationContext())
      }
      if (url.includes("/mcp-catalog/featured")) {
        return Response.json({ data: [], count: 0 })
      }
      // The second page is empty, as it is when a sync shortened the catalog.
      const skip = new URL(url, "http://localhost").searchParams.get("skip")
      return Response.json({ data: skip === "0" ? many : [], count: 60 })
    })
    const user = userEvent.setup()
    renderDialog()
    await screen.findByText("Server 0")

    await user.click(
      screen.getByRole("button", { name: "Next page, catalog servers" }),
    )

    expect(await screen.findByText(/Nothing on this page/)).toBeInTheDocument()
    expect(calls.some((url) => url.includes("skip=25&limit=25"))).toBe(true)
    // The way back is still there.
    await user.click(
      screen.getByRole("button", { name: "Previous page, catalog servers" }),
    )
    expect(await screen.findByText("Server 0")).toBeInTheDocument()
  })
})
