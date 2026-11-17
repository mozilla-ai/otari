import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { ApiKey, KeyScope } from "@/client"
import {
  holdersSentence,
  McpKeyScopeNotice,
} from "@/features/tools/McpKeyScopeNotice"
import { apiKey, organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const ALPHA = "11111111-1111-1111-1111-111111111111"

function key(id: string, scopes: KeyScope[], overrides: Partial<ApiKey> = {}) {
  return apiKey({ id, workspace_id: ALPHA, scopes, ...overrides })
}

function mockApi({
  keys = [] as ApiKey[],
  keysStatus,
  operator = true,
}: {
  keys?: ApiKey[]
  keysStatus?: number
  operator?: boolean
} = {}): string[] {
  const urls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    urls.push(url)
    if (url.includes("/keys?")) {
      if (keysStatus !== undefined) {
        return Response.json({ detail: "refused" }, { status: keysStatus })
      }
      return Response.json(keys)
    }
    return Response.json(organizationContext({ deployment_operator: operator }))
  })
  return urls
}

function renderNotice(
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
  return render(
    <QueryClientProvider client={client}>
      <McpKeyScopeNotice workspaceId={ALPHA} />
    </QueryClientProvider>,
    { wrapper: withRouter({ url: "/tools/mcp-servers" }) },
  )
}

describe("holdersSentence", () => {
  it("counts the workspace's keys for an operator and the caller's own otherwise", () => {
    expect(holdersSentence(1, 3, true)).toBe(
      "1 of the 3 keys in this workspace can call them.",
    )
    expect(holdersSentence(1, 3, false)).toBe(
      "1 of your 3 keys in this workspace can call them.",
    )
    expect(holdersSentence(0, 2, false)).toBe(
      "None of your 2 keys in this workspace can call them yet.",
    )
    expect(holdersSentence(0, 1, false)).toBe(
      "Your key in this workspace cannot call them yet.",
    )
    expect(holdersSentence(0, 1, true)).toBe(
      "The one key in this workspace cannot call them yet.",
    )
  })
})

describe("McpKeyScopeNotice", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("names both permissions, counts the keys that can call, and links to the keys page", async () => {
    const urls = mockApi({
      keys: [
        key("a", ["inference:run", "mcp:execute"]),
        key("b", ["inference:run"]),
        key("c", ["inference:run"]),
      ],
    })
    renderNotice()

    expect(
      await screen.findByText(
        /1 of the 3 keys in this workspace can call them/,
      ),
    ).toBeInTheDocument()
    expect(screen.getByText("mcp:execute")).toBeInTheDocument()
    expect(screen.getByText("mcp:read")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "API keys" })).toHaveAttribute(
      "href",
      "/keys",
    )
    // The operator reads the workspace's keys, not the whole deployment's.
    expect(urls.find((url) => url.includes("/keys?"))).toContain(
      `workspace_id=${ALPHA}`,
    )
  })

  it("says 'your keys' to someone who sees only their own", async () => {
    const urls = mockApi({
      operator: false,
      keys: [key("a", ["mcp:execute"]), key("b", ["inference:run"])],
    })
    renderNotice()

    expect(
      await screen.findByText(/1 of your 2 keys in this workspace/),
    ).toBeInTheDocument()
    expect(urls.some((url) => url.includes("/organizations/me/keys?"))).toBe(
      true,
    )
  })

  it("warns when no key can call the servers", async () => {
    mockApi({ keys: [key("a", ["inference:run"]), key("b", ["files:read"])] })
    const { container } = renderNotice()

    expect(
      await screen.findByText(/None of the 2 keys in this workspace/),
    ).toBeInTheDocument()
    expect(container.querySelector(".bg-danger")).not.toBeNull()
  })

  it("is gone once every usable key holds the scope", async () => {
    mockApi({
      keys: [
        key("a", ["inference:run", "mcp:execute"]),
        // Neither of these can authenticate, so neither is asked to change.
        key("b", ["inference:run"], { is_active: false }),
        key("c", ["inference:run"], { expires_at: "2020-01-01T00:00:00Z" }),
      ],
    })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const { container } = renderNotice(client)

    // Empty while the keys are still being read proves nothing, so wait for them.
    await vi.waitFor(() =>
      expect(
        client
          .getQueryCache()
          .findAll()
          .some(
            (query) =>
              query.queryKey.includes(ALPHA) &&
              query.state.status === "success",
          ),
      ).toBe(true),
    )
    expect(container).toBeEmptyDOMElement()
  })

  it("still says what keys need when there are none yet, without a count", async () => {
    mockApi({ keys: [] })
    renderNotice()

    expect(await screen.findByText("mcp:execute")).toBeInTheDocument()
    expect(screen.queryByText(/can call them/)).toBeNull()
  })

  it("drops the count when a later refresh of the keys fails", async () => {
    mockApi({ keys: [key("a", ["mcp:execute"]), key("b", ["inference:run"])] })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    renderNotice(client)
    expect(
      await screen.findByText(/1 of the 2 keys in this workspace/),
    ).toBeInTheDocument()

    vi.restoreAllMocks()
    mockApi({ keysStatus: 500 })
    await client.refetchQueries()

    await vi.waitFor(() =>
      expect(screen.queryByText(/can call them/)).toBeNull(),
    )
    expect(screen.getByText("mcp:execute")).toBeInTheDocument()
  })

  it("drops the count, not the notice, when the keys cannot be read", async () => {
    mockApi({ keysStatus: 403 })
    renderNotice()

    expect(await screen.findByText("mcp:execute")).toBeInTheDocument()
    expect(screen.queryByText(/can call them/)).toBeNull()
  })
})
