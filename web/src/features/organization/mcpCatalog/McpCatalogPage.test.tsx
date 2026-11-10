import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  McpCatalogFeatured,
  McpCatalogSyncStatus,
  OrganizationContext,
} from "@/client"
import { McpCatalogPage } from "@/features/organization/mcpCatalog/McpCatalogPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  mcpCatalogFeatured,
  mcpCatalogServer,
  mcpCatalogSyncStatus,
  organizationContext,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface Call {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

function mockApi({
  featured = [mcpCatalogFeatured()],
  sync = mcpCatalogSyncStatus(),
  context = {},
}: {
  featured?: McpCatalogFeatured[]
  sync?: McpCatalogSyncStatus
  context?: Partial<OrganizationContext>
} = {}): Call[] {
  const calls: Call[] = []
  let shelf = featured
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ url, method, body })
    if (url.includes("/mcp-catalog/featured")) {
      if (method === "DELETE") {
        shelf = shelf.filter((entry) => !url.endsWith(entry.id))
        return new Response(null, { status: 204 })
      }
      if (method === "POST") {
        return Response.json(mcpCatalogFeatured())
      }
      return Response.json({ data: shelf, count: shelf.length })
    }
    if (url.includes("/mcp-catalog/sync")) {
      return Response.json(
        method === "POST" ? { ...sync, requested: true } : sync,
        { status: method === "POST" ? 202 : 200 },
      )
    }
    if (url.includes("/mcp-catalog/servers")) {
      const servers = [
        mcpCatalogServer(),
        mcpCatalogServer({
          id: "88888888-8888-8888-8888-888888888888",
          name: "io.example/linear",
          title: "Linear",
        }),
      ]
      return Response.json({ data: servers, count: servers.length })
    }
    return Response.json(organizationContext(context))
  })
  return calls
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <McpCatalogPage />
      </QueryClientProvider>
    </DeploymentProvider>,
  )
}

const MEMBER = { role: "member", deployment_operator: false } as const
const ADMIN = { role: "admin", deployment_operator: false } as const

describe("McpCatalogPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists the shelf and says how each entry compares with the catalog", async () => {
    mockApi({
      context: ADMIN,
      featured: [
        mcpCatalogFeatured(),
        mcpCatalogFeatured({
          id: "a",
          name: "io.example/moved",
          title: "Moved",
          catalog_status: "changed",
          catalog_server_id: "server-moved",
        }),
        mcpCatalogFeatured({
          id: "b",
          name: "io.example/gone",
          title: "Gone",
          catalog_status: "removed",
          catalog_server_id: null,
          requires_token: true,
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Featured servers" })
    expect(await within(table).findByText("Linear")).toBeInTheDocument()
    expect(
      within(table).getByText("URL changed in catalog"),
    ).toBeInTheDocument()
    expect(within(table).getByText("No longer in catalog")).toBeInTheDocument()
    expect(within(table).getByText("Needs a token")).toBeInTheDocument()
    // Only the entry whose catalog URL moved can be updated to it.
    expect(
      within(table).getAllByRole("button", { name: /^Update / }),
    ).toHaveLength(1)
  })

  it("takes a changed entry's copy again from the catalog server it came from", async () => {
    const calls = mockApi({
      context: ADMIN,
      featured: [
        mcpCatalogFeatured({
          title: "Moved",
          catalog_status: "changed",
          catalog_server_id: "server-moved",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", {
        name: "Update Moved to the catalog's URL",
      }),
    )

    await waitFor(() =>
      expect(calls.find((call) => call.method === "POST")?.body).toEqual({
        server_id: "server-moved",
      }),
    )
  })

  it("removes an entry from the shelf", async () => {
    const calls = mockApi({ context: ADMIN })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Remove Linear" }),
    )

    expect(
      await screen.findByText("No server is featured yet."),
    ).toBeInTheDocument()
    expect(
      calls.some(
        (call) =>
          call.method === "DELETE" &&
          call.url.endsWith(`/mcp-catalog/featured/${mcpCatalogFeatured().id}`),
      ),
    ).toBe(true)
  })

  it("features a server picked from the catalog, and marks what is already there", async () => {
    const calls = mockApi({ context: ADMIN })
    const user = userEvent.setup()
    await renderPage()
    await screen.findByText("Linear")

    await user.click(screen.getByRole("button", { name: "Feature a server" }))
    const dialog = await screen.findByRole("dialog", {
      name: "Feature a server",
    })
    // Linear is on the shelf already, so only Sentry is offered.
    expect(await within(dialog).findByText("Featured")).toBeInTheDocument()
    expect(
      within(dialog).queryByRole("button", { name: "Feature Linear" }),
    ).not.toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Feature Sentry" }),
    )

    await waitFor(() =>
      expect(calls.find((call) => call.method === "POST")?.body).toEqual({
        server_id: mcpCatalogServer().id,
      }),
    )
  })

  it("shows a member the shelf and nothing that changes it", async () => {
    const calls = mockApi({ context: MEMBER })
    await renderPage()

    expect(await screen.findByText("Linear")).toBeInTheDocument()
    expect(
      await screen.findByText(/Only an owner or admin of the organization/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Feature a server" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Remove Linear" }),
    ).not.toBeInTheDocument()
    // The sync is the operator's: it is neither shown nor asked for.
    expect(screen.queryByText("Registry sync")).not.toBeInTheDocument()
    expect(calls.some((call) => call.url.includes("/mcp-catalog/sync"))).toBe(
      false,
    )
  })

  it("shows an operator what the last sync did and starts one on request", async () => {
    const calls = mockApi({
      sync: mcpCatalogSyncStatus({
        outcome: "partial",
        error: "the registry answered HTTP 502",
        finished_at: "2026-11-10T09:05:00Z",
        pages: 12,
        fetched: 1200,
        kept: 1000,
        server_count: 4321,
      }),
    })
    const user = userEvent.setup()
    await renderPage()

    expect(
      await screen.findByText(/Partial on .*the registry answered HTTP 502/),
    ).toBeInTheDocument()
    expect(screen.getByText(/nothing was removed/)).toBeInTheDocument()
    expect(screen.getByText(/4,321 servers in the catalog/)).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Sync now" }))

    expect(
      await screen.findByText("A sync starts within a minute."),
    ).toBeInTheDocument()
    expect(
      calls.some(
        (call) =>
          call.method === "POST" && call.url.endsWith("/mcp-catalog/sync"),
      ),
    ).toBe(true)
    // One is on its way, so a second press has nothing to add.
    expect(screen.getByRole("button", { name: "Sync now" })).toBeDisabled()
  })

  it("does not offer a sync on a deployment that switched it off", async () => {
    mockApi({ sync: mcpCatalogSyncStatus({ mode: "off" }) })
    await renderPage()

    expect(
      await screen.findByText(/never contacts the MCP Registry/),
    ).toBeInTheDocument()
    expect(screen.getByText("No sync has run yet.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Sync now" })).toBeDisabled()
  })
})
