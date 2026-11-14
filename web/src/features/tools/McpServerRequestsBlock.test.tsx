import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { McpServerRequester, McpServerRequestGroup } from "@/client"
import {
  McpServerRequestsBlock,
  pickFor,
  whoAsked,
} from "@/features/tools/McpServerRequestsBlock"
import type { CatalogPick } from "@/features/tools/mcpCatalog"
import { mcpCatalogServer } from "@/tests/fixtures"

const WORKSPACE = "11111111-1111-1111-1111-111111111111"
const SENTRY = mcpCatalogServer({ publisher_kind: "vendor" })

function requester(
  name: string,
  overrides: Partial<McpServerRequester> = {},
): McpServerRequester {
  return {
    request_id: `request-${name}`,
    user_id: `user-${name}`,
    full_name: name,
    email: `${name.toLowerCase()}@example.com`,
    note: null,
    requested_at: "2026-11-10T09:00:00+00:00",
    ...overrides,
  }
}

function group(
  overrides: Partial<McpServerRequestGroup> = {},
): McpServerRequestGroup {
  return {
    catalog_name: SENTRY.name,
    title: SENTRY.title,
    url: SENTRY.url,
    catalog_server: SENTRY,
    requesters: [
      requester("Ada", { note: "Triage production errors" }),
      requester("Grace"),
    ],
    requester_count: 2,
    first_requested_at: "2026-11-10T09:00:00+00:00",
    ...overrides,
  }
}

interface Call {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

function mockApi(groups: McpServerRequestGroup[]): Call[] {
  const calls: Call[] = []
  let waiting = [...groups]
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ url, method, body })
    if (url.endsWith("/decline")) {
      waiting = waiting.filter(
        (entry) => entry.catalog_name !== body?.catalog_name,
      )
      return Response.json({ declined: 2 })
    }
    const params = new URL(url, "http://localhost").searchParams
    const skip = Number(params.get("skip") ?? 0)
    const limit = Number(params.get("limit") ?? 100)
    return Response.json({
      data: waiting.slice(skip, skip + limit),
      count: waiting.reduce(
        (total, entry) => total + entry.requesters.length,
        0,
      ),
      server_count: waiting.length,
    })
  })
  return calls
}

function renderBlock(onAdd: (pick: CatalogPick) => void = () => {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <McpServerRequestsBlock workspaceId={WORKSPACE} onAdd={onAdd} />
    </QueryClientProvider>,
  )
}

describe("McpServerRequestsBlock", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("is absent while nobody is waiting", async () => {
    const calls = mockApi([])
    renderBlock()

    await waitFor(() => expect(calls).toHaveLength(1))
    expect(
      screen.queryByRole("heading", { name: /Requests/ }),
    ).not.toBeInTheDocument()
  })

  it("says who asked for which server, how many, and why", async () => {
    mockApi([group()])
    renderBlock()

    // The heading carries the count of people waiting.
    expect(
      await screen.findByRole("heading", { name: /^Requests\s*\(2\)$/ }),
    ).toBeVisible()
    const row = screen.getByRole("region", { name: "Requests for Sentry" })
    expect(within(row).getByText("Sentry")).toBeVisible()
    expect(within(row).getByText("Official")).toBeVisible()
    expect(within(row).getByText(SENTRY.url)).toBeVisible()
    expect(within(row).getByText(/^Asked by Ada and Grace on /)).toBeVisible()
    expect(within(row).getByText("Triage production errors")).toBeVisible()
  })

  it("hands Add the catalog's entry for the server", async () => {
    mockApi([group()])
    const onAdd = vi.fn()
    renderBlock(onAdd)

    await userEvent.click(
      await screen.findByRole("button", { name: "Add Sentry" }),
    )

    expect(onAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        name: SENTRY.name,
        title: "Sentry",
        url: SENTRY.url,
        isOfficial: true,
      }),
    )
  })

  it("declines everyone waiting for a server with the reason typed", async () => {
    const calls = mockApi([group()])
    const user = userEvent.setup()
    renderBlock()

    await user.click(
      await screen.findByRole("button", { name: "Decline Sentry" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByText(
        "Ada and Grace will see that the request was declined.",
      ),
    ).toBeVisible()
    await user.type(
      within(dialog).getByLabelText("Reason"),
      "We use Datadog for this",
    )
    await user.click(within(dialog).getByRole("button", { name: "Decline" }))

    await waitFor(() =>
      expect(calls.find((call) => call.method === "POST")).toMatchObject({
        url: expect.stringMatching(
          new RegExp(`/workspaces/${WORKSPACE}/mcp-server-requests/decline$`),
        ),
        body: { catalog_name: SENTRY.name, reason: "We use Datadog for this" },
      }),
    )
    // The list is read again, and with nobody waiting the block goes.
    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: /Requests/ }),
      ).not.toBeInTheDocument(),
    )
  })

  it("pages by server, so a server past the first page can be answered", async () => {
    const servers = Array.from({ length: 26 }, (_, index) =>
      group({ catalog_name: `server-${index}`, title: `Server ${index}` }),
    )
    const calls = mockApi(servers)
    const user = userEvent.setup()
    renderBlock()

    expect(
      await screen.findByRole("region", { name: "Requests for Server 0" }),
    ).toBeVisible()
    expect(
      screen.getAllByRole("region", { name: /^Requests for / }),
    ).toHaveLength(25)
    // The heading still counts people waiting; the pager counts servers.
    expect(
      screen.getByRole("heading", { name: /^Requests\s*\(52\)$/ }),
    ).toBeVisible()
    expect(screen.getByText("1–25 of 26")).toBeVisible()

    await user.click(screen.getByRole("button", { name: "Next page" }))

    expect(
      await screen.findByRole("region", { name: "Requests for Server 25" }),
    ).toBeVisible()
    expect(
      screen.getAllByRole("region", { name: /^Requests for / }),
    ).toHaveLength(1)
    expect(calls.at(-1)?.url).toMatch(/\?skip=25&limit=25$/)
  })

  it("has no pager while every server fits on one page", async () => {
    mockApi([group()])
    renderBlock()

    await screen.findByRole("region", { name: "Requests for Sentry" })
    expect(
      screen.queryByRole("button", { name: "Next page" }),
    ).not.toBeInTheDocument()
  })

  it("names the people who asked without listing a crowd", () => {
    const people = ["Ada", "Grace", "Alan", "Edsger"].map((name) =>
      requester(name),
    )
    expect(whoAsked(people.slice(0, 1))).toBe("Ada")
    expect(whoAsked(people.slice(0, 2))).toBe("Ada and Grace")
    expect(whoAsked(people.slice(0, 3))).toBe("Ada, Grace and 1 other")
    expect(whoAsked(people)).toBe("Ada, Grace and 2 others")
    // The gateway lists the first few of a server's requesters, and counts them all.
    expect(whoAsked(people, 9)).toBe("Ada, Grace and 7 others")
    expect(
      whoAsked([requester("x", { full_name: null, email: "x@example.com" })]),
    ).toBe("x@example.com")
  })

  it("still leads to an add once the catalog has dropped the server", () => {
    expect(pickFor(group({ catalog_server: null }))).toEqual({
      name: SENTRY.name,
      title: "Sentry",
      description: "",
      url: SENTRY.url,
      requiresToken: false,
      logoUrl: "",
      logoTone: null,
      isOfficial: false,
    })
  })
})
