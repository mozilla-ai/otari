import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, OrgWebSearchKey } from "@/client"
import { OrganizationWebSearchKeysPage } from "@/features/organization/webSearchKeys/OrganizationWebSearchKeysPage"
import { organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

function key(overrides: Partial<OrgWebSearchKey> = {}): OrgWebSearchKey {
  return {
    id: "77777777-7777-7777-7777-777777777777",
    organization_id: "11111111-1111-1111-1111-111111111111",
    kind: "search",
    provider: "exa",
    name: "AAA",
    last4: "a1b2",
    api_base: null,
    options: null,
    is_org_default: false,
    usable: true,
    archived_at: null,
    created_at: "2026-10-28T00:00:00+00:00",
    updated_at: null,
    ...overrides,
  }
}

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    keys?: OrgWebSearchKey[]
    context?: OrganizationContext
    count?: number
  } = {},
) {
  const keys = opts.keys ?? []
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (url.includes("/web-search-keys")) {
      if (method === "GET") {
        return jsonResponse({ count: opts.count ?? keys.length, data: keys })
      }
      if (method === "DELETE" && !url.endsWith("/default")) {
        return new Response(null, { status: 204 })
      }
      return jsonResponse(keys[0] ?? key())
    }
    return jsonResponse(opts.context ?? organizationContext())
  })
  return requests
}

function writes(requests: Request[], method: string, path: string) {
  return requests.filter(
    (request) => request.method === method && request.url.endsWith(path),
  )
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <OrganizationWebSearchKeysPage />
    </QueryClientProvider>,
    { url: "/organization/web-search-keys" },
  )
}

describe("OrganizationWebSearchKeysPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists each kind's keys under its own heading, with what a workspace without a pin uses", async () => {
    mockApi({
      keys: [
        key({ name: "M", provider: "mojeek", is_org_default: true }),
        key({
          id: "88888888-8888-8888-8888-888888888888",
          name: "BBB",
          last4: "c3d4",
          api_base: "https://search.acme.example",
          usable: false,
        }),
        key({
          id: "99999999-9999-9999-9999-999999999999",
          kind: "fetch",
          provider: "tabstack",
          name: "main",
          last4: "77c1",
        }),
      ],
    })
    await renderPage()

    const search = await screen.findByRole("grid", { name: "Search providers" })
    expect(await within(search).findByText("M")).toBeInTheDocument()
    expect(within(search).getByText("WORKSPACE DEFAULT")).toBeInTheDocument()
    expect(within(search).getByText("UNREADABLE")).toBeInTheDocument()
    expect(
      within(search).getByText("https://search.acme.example"),
    ).toBeInTheDocument()
    expect(
      screen.getByText("Every workspace that pinned no key uses M."),
    ).toBeInTheDocument()
    const fetch = screen.getByRole("grid", { name: "Fetch providers" })
    expect(within(fetch).getByText("••••77c1")).toBeInTheDocument()
    expect(
      screen.getByText(
        /With no workspace default, workspaces fetch with this deployment's own tool/,
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/One key can't be decrypted on this deployment/),
    ).toBeInTheDocument()
  })

  it("says workspaces without a pin use this deployment's tool when the default can't be decrypted", async () => {
    mockApi({ keys: [key({ name: "M", is_org_default: true, usable: false })] })
    await renderPage()

    expect(
      await screen.findByText(
        "M can't be decrypted on this deployment, so workspaces that pinned no key use this deployment's own tool.",
      ),
    ).toBeInTheDocument()
  })

  it("says when the organization has more keys than one page lists", async () => {
    const requests = mockApi({ keys: [key()], count: 1200 })
    await renderPage()

    expect(
      await screen.findByText(
        "This page lists the first 1,000 of this organization's 1,200 keys.",
      ),
    ).toBeInTheDocument()
    expect(
      requests.find((request) => request.url.includes("/web-search-keys"))?.url,
    ).toContain("include_archived=true&limit=1000")
  })

  it("sets a key as workspace default, and unsets it", async () => {
    const user = userEvent.setup()
    const requests = mockApi({
      keys: [
        key(),
        key({
          id: "88888888-8888-8888-8888-888888888888",
          name: "M",
          provider: "mojeek",
          is_org_default: true,
        }),
      ],
    })
    await renderPage()

    await user.click(
      await screen.findByRole("button", {
        name: "Set AAA as workspace default",
      }),
    )
    await user.click(
      screen.getByRole("button", { name: "Unset M as workspace default" }),
    )

    await waitFor(() =>
      expect(
        writes(
          requests,
          "POST",
          "/web-search-keys/77777777-7777-7777-7777-777777777777/default",
        ),
      ).toHaveLength(1),
    )
    await waitFor(() =>
      expect(
        writes(
          requests,
          "DELETE",
          "/web-search-keys/88888888-8888-8888-8888-888888888888/default",
        ),
      ).toHaveLength(1),
    )
  })

  it("archives a key after a confirmation, and deletes an archived one after another", async () => {
    const user = userEvent.setup()
    const requests = mockApi({
      keys: [
        key(),
        key({
          id: "88888888-8888-8888-8888-888888888888",
          name: "gone",
          archived_at: "2026-10-29T00:00:00+00:00",
        }),
      ],
    })
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Archive" }))
    await user.click(screen.getByRole("button", { name: "Archive" }))
    await waitFor(() =>
      expect(
        writes(
          requests,
          "POST",
          "/web-search-keys/77777777-7777-7777-7777-777777777777/archive",
        ),
      ).toHaveLength(1),
    )

    await user.click(screen.getByRole("checkbox", { name: /Show archived/ }))
    await user.click(await screen.findByRole("button", { name: "Delete gone" }))
    await user.click(
      await screen.findByRole("button", { name: "Delete permanently" }),
    )
    await waitFor(() =>
      expect(
        writes(
          requests,
          "DELETE",
          "/web-search-keys/88888888-8888-8888-8888-888888888888",
        ),
      ).toHaveLength(1),
    )
  })

  it("withholds the keys from a member", async () => {
    const requests = mockApi({
      context: organizationContext({ role: "member" }),
    })
    await renderPage()

    expect(
      await screen.findByText(/Only organization owners and admins can see/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("grid")).toBeNull()
    expect(
      requests.some((request) => request.url.includes("/web-search-keys")),
    ).toBe(false)
  })

  it("warns when the gateway cannot encrypt a key", async () => {
    mockApi({
      context: organizationContext({
        provider_key_encryption_available: false,
      }),
    })
    await renderPage()

    expect(
      await screen.findByText(/is not set, so keys can.t be encrypted/),
    ).toBeInTheDocument()
  })
})
