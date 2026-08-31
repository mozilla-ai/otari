import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, OrgProviderKey } from "@/client"
import { OrganizationProviderKeysPage } from "@/features/organization/OrganizationProviderKeysPage"
import { organizationContext, orgProviderKey } from "@/tests/fixtures"

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

interface MockOpts {
  keys?: OrgProviderKey[]
  context?: OrganizationContext
}

function mockApi(opts: MockOpts = {}) {
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

    if (url.includes("/provider-keys")) {
      if (method === "GET") {
        return jsonResponse({ count: keys.length, data: keys })
      }
      // Every write answers with a key-shaped body the page only re-reads
      // through the invalidated list, so one row is enough for all of them.
      return jsonResponse(keys[0] ?? orgProviderKey())
    }
    // What the deployment actually answers this page's audience: `/v1/settings`
    // is operator-only and an organization owner is not one. Mocked as the
    // refusal rather than as a body, so a page that went back to reading it
    // would fail here rather than pass on a fixture no tenant ever receives.
    if (url.includes("/v1/settings")) {
      return jsonResponse({ detail: "Not authorized" }, 403)
    }
    if (url.includes("/v1/providers/catalog")) {
      return jsonResponse([{ id: "anthropic", name: "Anthropic" }])
    }
    return jsonResponse(opts.context ?? organizationContext())
  })
  return requests
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationProviderKeysPage", () => {
  it("lists the organization's keys with the default marked", async () => {
    mockApi({
      keys: [
        orgProviderKey({ name: "Production", is_org_default: true }),
        orgProviderKey({
          id: "77777777-7777-7777-7777-777777777777",
          name: "Staging",
          provider: "anthropic",
          api_base: "https://proxy.example.com/v1",
        }),
      ],
    })
    renderPage(<OrganizationProviderKeysPage />)

    expect(await screen.findByText("Production")).toBeInTheDocument()
    expect(screen.getByText("default")).toBeInTheDocument()
    expect(screen.getByText("Staging")).toBeInTheDocument()
    expect(screen.getByText("anthropic")).toBeInTheDocument()
    expect(screen.getByText("https://proxy.example.com/v1")).toBeInTheDocument()
    // The credential itself never comes back, so the page can only ever show
    // the tail the API publishes.
    expect(screen.getAllByText("••••abcd").length).toBeGreaterThan(0)
  })

  it("reads the organization's own keys, not the deployment's credentials", async () => {
    // The whole reason this page exists: `/v1/provider-credentials` is keyed on
    // an instance name and belongs to the process, so a page that read it would
    // be showing every tenant the same rows.
    const requests = mockApi({ keys: [orgProviderKey()] })
    renderPage(<OrganizationProviderKeysPage />)

    await screen.findByText("Production")
    expect(
      requests.some((request) =>
        request.url.includes("/v1/organizations/me/provider-keys"),
      ),
    ).toBe(true)
    expect(
      requests.some((request) =>
        request.url.includes("/v1/provider-credentials"),
      ),
    ).toBe(false)
  })

  it("creates a key for the provider that was picked", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationProviderKeysPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(screen.getByRole("textbox", { name: /Name/ }), "Production")
    await user.type(screen.getByLabelText("API key"), "sk-secret")
    await user.click(
      screen.getByRole("button", { name: "Add provider key", hidden: false }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url.endsWith("/v1/organizations/me/provider-keys"),
        ),
      ).toBe(true)
    })
    const post = requests.find(
      (request) =>
        request.method === "POST" &&
        request.url.endsWith("/v1/organizations/me/provider-keys"),
    )
    expect(post?.body).toMatchObject({
      provider: "anthropic",
      name: "Production",
      api_key: "sk-secret",
    })
  })

  it("keeps the stored credential when an edit leaves the key box blank", async () => {
    // The box is never prefilled, because the gateway returns no plaintext to
    // prefill it with. Sending it as an explicit null would clear the key the
    // operator did not touch.
    const requests = mockApi({ keys: [orgProviderKey()] })
    const user = userEvent.setup()
    renderPage(<OrganizationProviderKeysPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))
    await user.click(await screen.findByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(requests.some((request) => request.method === "PATCH")).toBe(true)
    })
    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.body).not.toHaveProperty("api_key")
  })

  it("makes a key the organization default", async () => {
    const requests = mockApi({ keys: [orgProviderKey()] })
    const user = userEvent.setup()
    renderPage(<OrganizationProviderKeysPage />)

    await user.click(
      await screen.findByRole("button", { name: "Make default" }),
    )

    await waitFor(() => {
      expect(requests.some((request) => request.url.endsWith("/default"))).toBe(
        true,
      )
    })
  })

  it("hides archived keys until they are asked for, and offers delete only there", async () => {
    mockApi({
      keys: [
        orgProviderKey(),
        orgProviderKey({
          id: "88888888-8888-8888-8888-888888888888",
          name: "Retired",
          archived_at: "2026-08-20T00:00:00+00:00",
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<OrganizationProviderKeysPage />)

    await screen.findByText("Production")
    expect(screen.queryByText("Retired")).toBeNull()
    // Delete is permanent and the API accepts it for an archived key alone, so
    // it is never offered beside a live one.
    expect(screen.queryByRole("button", { name: "Delete" })).toBeNull()

    await user.click(screen.getByText("Show archived (1)"))

    expect(await screen.findByText("Retired")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Restore" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument()
  })

  it("withholds the keys and their read from a member who cannot manage the organization", async () => {
    // The list read is organization owner/admin-gated on the server
    // (otari-ai#1944), so a member reaching this URL is answered 403. The read
    // is not made, and the table goes with it: an empty one would say the
    // organization has no keys rather than that they cannot see them.
    const requests = mockApi({
      keys: [orgProviderKey()],
      context: organizationContext({ role: "member" }),
    })
    renderPage(<OrganizationProviderKeysPage />)

    expect(
      await screen.findByText(/Only organization owners and admins/),
    ).toBeInTheDocument()
    expect(screen.queryByText("Production")).toBeNull()
    // HeroUI's table is a `grid`, so this is the table itself being absent and
    // not merely empty of the row above.
    expect(
      screen.queryByRole("grid", { name: "Organization provider keys" }),
    ).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Add provider key" }),
    ).toBeNull()
    expect(screen.queryByRole("button", { name: "Edit" })).toBeNull()
    expect(
      requests.some((request) =>
        request.url.includes("/v1/organizations/me/provider-keys"),
      ),
    ).toBe(false)
  })

  it("disables adding a key when the deployment cannot encrypt one", async () => {
    // Same gate the deployment-wide providers page applies: without
    // OTARI_SECRET_KEY the write would fail at submit time.
    mockApi({
      context: organizationContext({
        provider_key_encryption_available: false,
      }),
    })
    renderPage(<OrganizationProviderKeysPage />)

    expect(
      await screen.findByRole("button", { name: "Add provider key" }),
    ).toBeDisabled()
    expect(screen.getByText(/OTARI_SECRET_KEY/)).toBeInTheDocument()
  })

  it("keeps adding available for an owner the operator-only settings read refuses", async () => {
    // The bug this page shipped with (#839): the flag was inferred from
    // `/v1/settings`, which 403s for every organization owner, so the banner
    // reported a missing key on a deployment where the write path works.
    const requests = mockApi()
    renderPage(<OrganizationProviderKeysPage />)

    expect(
      await screen.findByRole("button", { name: "Add provider key" }),
    ).toBeEnabled()
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
    expect(
      requests.some((request) => request.url.includes("/v1/settings")),
    ).toBe(false)
  })

  it("says nothing about the key when the context read is what failed", async () => {
    // The shape the original bug had, from the other direction: with no context
    // the encryption state is unknown, so the page must report the read that
    // failed rather than claim the key is missing. It behaves correctly today
    // only because `canManage(undefined)` is false and the banner is gated on
    // it, which is a role check standing in for an encryption one; this pins the
    // outcome so a later change to either gate cannot quietly restore the lie.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).includes("/provider-keys")
        ? jsonResponse({ count: 0, data: [] })
        : jsonResponse({ detail: "Tenancy is unavailable" }, 500),
    )
    renderPage(<OrganizationProviderKeysPage />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tenancy is unavailable",
    )
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Add provider key" }),
    ).toBeNull()
    // Nor is the role claimed either way: with the context unread the page
    // cannot say the caller is a member, so the refusal banner stays off and the
    // table, whose read is gated on that same role, is not drawn empty.
    expect(screen.queryByText(/Only organization owners and admins/)).toBeNull()
    expect(
      screen.queryByRole("grid", { name: "Organization provider keys" }),
    ).toBeNull()
  })

  it("reports a list that could not be read instead of an empty table", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).includes("/provider-keys")
        ? jsonResponse({ detail: "Tenancy is unavailable" }, 500)
        : jsonResponse(organizationContext()),
    )
    renderPage(<OrganizationProviderKeysPage />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tenancy is unavailable",
    )
  })
})
