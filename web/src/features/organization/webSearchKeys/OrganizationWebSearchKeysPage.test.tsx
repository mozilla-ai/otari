import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  OrgWebSearchKey,
  SearchProviderInfo,
} from "@/client"
import { OrganizationWebSearchKeysPage } from "@/features/organization/webSearchKeys/OrganizationWebSearchKeysPage"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"
import { pickOption } from "@/tests/select"

const KEYS = `${API_ROOT}/organizations/me/web-search-keys`

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

const SEARCH_CATALOG: SearchProviderInfo[] = [
  {
    id: "exa",
    kind: "search",
    requires_api_key: true,
    requires_api_base: false,
    default_api_base: "https://api.exa.ai",
    tier: "production",
    options: [
      {
        name: "type",
        type: "string",
        enum: ["auto", "fast"],
        default: "auto",
        description: "How Exa searches.",
        operator_only: false,
      },
    ],
  },
  {
    id: "mojeek",
    kind: "search",
    requires_api_key: true,
    requires_api_base: false,
    default_api_base: "https://api.mojeek.com",
    tier: "production",
    options: [],
  },
  // As the catalog lists it to an operator: with the endpoint the
  // deployment's own searxng tools inherit, which a key never does.
  {
    id: "searxng",
    kind: "search",
    requires_api_key: false,
    requires_api_base: true,
    default_api_base: "http://searxng:8080",
    tier: "production",
    options: [],
  },
  // Listed with no tier, as the catalog lists a provider no library serves.
  {
    id: "legacy",
    kind: "search",
    requires_api_key: false,
    requires_api_base: false,
    tier: null,
    options: null,
  },
]

const FETCH_CATALOG: SearchProviderInfo[] = [
  {
    id: "tabstack",
    kind: "fetch",
    requires_api_key: true,
    requires_api_base: false,
    tier: "production",
    options: [],
    reports_final_url: false,
  },
]

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
    patchFails?: boolean
    catalogFails?: boolean
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
    if (url.includes("/search-tools/providers")) {
      if (opts.catalogFails) {
        return jsonResponse(
          { detail: "The provider catalog is unavailable." },
          503,
        )
      }
      return jsonResponse(
        url.includes("kind=fetch") ? FETCH_CATALOG : SEARCH_CATALOG,
      )
    }
    if (url.includes("/web-search-keys")) {
      if (method === "GET") {
        return jsonResponse({ count: opts.count ?? keys.length, data: keys })
      }
      if (method === "PATCH" && opts.patchFails) {
        return jsonResponse(
          {
            detail:
              "Moving this key to another api_base needs its api_key in the same request.",
          },
          400,
        )
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

  it("adds a key on a provider the catalog offers, leaving out one no library serves", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Add search provider" }),
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("button", { name: /Provider/ }))
    expect(screen.getByRole("option", { name: "exa" })).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "legacy" })).toBeNull()
    await user.click(screen.getByRole("option", { name: "exa" }))
    await user.type(within(dialog).getByLabelText(/^Name/), "AAA")
    await user.type(within(dialog).getByLabelText(/API key/), "exa-secret")
    await user.click(
      within(dialog).getByRole("button", { name: "More settings for exa" }),
    )
    await pickOption(user, "type", "fast", dialog)
    await user.click(
      within(dialog).getByRole("button", { name: "Add search provider" }),
    )

    await waitFor(() => expect(writes(requests, "POST", KEYS)).toHaveLength(1))
    expect(writes(requests, "POST", KEYS)[0].body).toEqual({
      kind: "search",
      provider: "exa",
      name: "AAA",
      api_key: "exa-secret",
      api_base: null,
      options: { type: "fast" },
    })
  })

  it("asks for its own endpoint for a key on a provider that requires one, never the deployment's", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Add search provider" }),
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("button", { name: /Provider/ }))
    await user.click(screen.getByRole("option", { name: "searxng" }))
    await user.type(within(dialog).getByLabelText(/^Name/), "Team SearXNG")
    const endpoint = within(dialog).getByLabelText(/API base URL/)
    expect(endpoint).toHaveAttribute("placeholder", "https://…")
    expect(within(dialog).queryByText(/searxng:8080/)).toBeNull()
    expect(
      within(dialog).getByText(
        "This provider has no endpoint of its own, so it needs one here.",
      ),
    ).toBeInTheDocument()
    const add = within(dialog).getByRole("button", {
      name: "Add search provider",
    })
    expect(add).toBeDisabled()
    await user.type(endpoint, "https://searx.acme.example")
    await user.click(add)

    await waitFor(() => expect(writes(requests, "POST", KEYS)).toHaveLength(1))
    expect(writes(requests, "POST", KEYS)[0].body).toEqual({
      kind: "search",
      provider: "searxng",
      name: "Team SearXNG",
      api_key: null,
      api_base: "https://searx.acme.example",
      options: null,
    })
  })

  it("edits a key by sending only what changed, and asks for the key again for a new URL", async () => {
    const user = userEvent.setup()
    const requests = mockApi({ keys: [key({ options: { type: "auto" } })] })
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit AAA" }))
    const dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: "More settings for exa" }),
    )
    await user.type(
      within(dialog).getByLabelText(/API base URL/),
      "https://search.acme.example",
    )
    expect(
      within(dialog).getByText(
        "Enter the API key again: it goes to the new URL.",
      ),
    ).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeDisabled()
    await user.type(within(dialog).getByLabelText(/API key/), "exa-rotated")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    const path =
      "/organizations/me/web-search-keys/77777777-7777-7777-7777-777777777777"
    await waitFor(() => expect(writes(requests, "PATCH", path)).toHaveLength(1))
    expect(writes(requests, "PATCH", path)[0].body).toEqual({
      api_key: "exa-rotated",
      api_base: "https://search.acme.example",
    })
  })

  it("shows the gateway's refusal in the dialog", async () => {
    const user = userEvent.setup()
    mockApi({ keys: [key()], patchFails: true })
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit AAA" }))
    const dialog = await screen.findByRole("dialog")
    await user.clear(within(dialog).getByLabelText(/^Name/))
    await user.type(within(dialog).getByLabelText(/^Name/), "renamed")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(
      await within(dialog).findByText(/needs its api_key in the same request/),
    ).toBeInTheDocument()
  })

  it("keeps adding and editing closed, and says why, while the provider catalog cannot load", async () => {
    mockApi({ keys: [key()], catalogFails: true })
    await renderPage()

    expect(
      await screen.findByText("The provider catalog is unavailable."),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Add search provider" }),
    ).toBeDisabled()
    expect(screen.getByRole("button", { name: "Edit AAA" })).toBeDisabled()
  })

  it("edits only the name and secret of a key on a provider the catalog does not list", async () => {
    const user = userEvent.setup()
    mockApi({ keys: [key({ provider: "tavily", name: "old" })] })
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit old" }))
    const dialog = await screen.findByRole("dialog")

    expect(
      within(dialog).getByText(/runs on the older search code/),
    ).toBeInTheDocument()
    expect(
      within(dialog).queryByRole("button", { name: /More settings/ }),
    ).toBeNull()
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

  it("disables adding when the gateway cannot encrypt a key", async () => {
    mockApi({
      context: organizationContext({
        provider_key_encryption_available: false,
      }),
    })
    await renderPage()

    expect(
      await screen.findByText(/is not set, so keys can.t be encrypted/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Add search provider" }),
    ).toBeDisabled()
  })
})
