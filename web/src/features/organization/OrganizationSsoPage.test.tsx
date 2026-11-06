import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  OrganizationMember,
  OrganizationSso,
  OrganizationSsoPreflight,
} from "@/client"
import { OrganizationSsoPage } from "@/features/organization/OrganizationSsoPage"
import { formatDate } from "@/shared/helpers/format"
import {
  oidcConnection,
  oidcConnectionDomain,
  organizationContext,
  organizationMember,
  organizationSso,
  organizationSsoPreflight,
  organizationSsoPreflightMember,
} from "@/tests/fixtures"

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

interface MockOpts {
  sso?: OrganizationSso
  context?: OrganizationContext
  members?: OrganizationMember[]
  preflight?: OrganizationSsoPreflight
  /** What a save answers, so a test can drive the refusal path. */
  save?: () => Response
  /** What setting enforcement answers, so a test can drive each of its refusals. */
  enforcement?: () => Response
}

function mockApi(opts: MockOpts = {}) {
  let served = opts.sso ?? organizationSso()
  const members = opts.members ?? []
  const preflight = opts.preflight ?? organizationSsoPreflight()
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/sso/test")) {
      return jsonResponse(served)
    }
    if (url.includes("/sso/preflight")) {
      return jsonResponse(preflight)
    }
    if (url.includes("/sso/enforcement")) {
      return opts.enforcement
        ? opts.enforcement()
        : jsonResponse(undefined, 204)
    }
    if (url.includes("/sso")) {
      if (method === "GET") return jsonResponse(served)
      if (method === "PUT") {
        if (opts.save) return opts.save()
        // The served connection reflects the write, so a page that re-reads after saving sees the
        // stored connection rather than none, which is what a real gateway answers.
        const body = JSON.parse(String(init?.body ?? "{}"))
        served = {
          ...served,
          connection: oidcConnection({
            issuer: body.issuer,
            resolved_issuer: body.issuer,
            client_id: body.client_id,
            client_secret_last4: String(body.client_secret ?? "").slice(-4),
            display_name: body.display_name,
            enabled: body.enabled,
          }),
        }
        return jsonResponse(served)
      }
      if (method === "DELETE") return jsonResponse(undefined, 204)
    }
    if (url.includes("/organizations/me/members")) {
      return jsonResponse({ data: members, count: members.length })
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

describe("OrganizationSsoPage", () => {
  it("shows the redirect URI as one copyable value", async () => {
    mockApi()
    renderPage(<OrganizationSsoPage />)

    expect(await screen.findByLabelText("Redirect URI")).toHaveValue(
      "https://otari.example.com/auth/sso/callback",
    )
  })

  it("shows the redirect URI before any connection is saved", async () => {
    mockApi({ sso: organizationSso({ connection: null }) })
    renderPage(<OrganizationSsoPage />)

    expect(await screen.findByLabelText("Redirect URI")).toHaveValue(
      "https://otari.example.com/auth/sso/callback",
    )
    expect(
      screen.getByRole("button", { name: "Add connection" }),
    ).toBeInTheDocument()
  })

  it("names each verified domain and when its proof expires", async () => {
    mockApi({
      sso: organizationSso({
        connection: oidcConnection({
          domains: [
            oidcConnectionDomain({
              domain: "acme.example",
              proof_expires_at: "2099-01-01T00:00:00+00:00",
            }),
          ],
        }),
      }),
    })
    renderPage(<OrganizationSsoPage />)

    expect(await screen.findByText("acme.example")).toBeInTheDocument()
    expect(
      screen.getByText(
        `proof expires ${formatDate("2099-01-01T00:00:00+00:00")}`,
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/nobody can sign in here until one is re-verified/),
    ).toBeInTheDocument()
  })

  it("says the email_verified claim is not trusted", async () => {
    mockApi()
    renderPage(<OrganizationSsoPage />)

    expect(await screen.findByText("email_verified")).toBeInTheDocument()
    expect(screen.getByText(/does not trust/)).toBeInTheDocument()
  })

  it("never renders the client secret", async () => {
    mockApi({ sso: organizationSso({ connection: oidcConnection() }) })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await screen.findByText("Acme Okta")
    // Only the last four characters ever reach the page.
    expect(screen.getByText("••••wxyz")).toBeInTheDocument()

    await user.click(
      await screen.findByRole("button", { name: "Edit connection" }),
    )
    expect(await screen.findByLabelText("Client secret")).toHaveValue("")
  })

  it("withholds management from a non-admin member, and reads nothing", async () => {
    const requests = mockApi({
      context: organizationContext({ role: "member" }),
    })
    renderPage(<OrganizationSsoPage />)

    expect(
      await screen.findByText(
        "Only organization owners and admins can manage single sign-on.",
      ),
    ).toBeInTheDocument()
    expect(requests.some((request) => request.url.includes("/sso"))).toBe(false)
  })

  it("says what to configure when there is no address to register", async () => {
    mockApi({ sso: organizationSso({ connection: null, redirect_uri: null }) })
    renderPage(<OrganizationSsoPage />)

    expect(
      await screen.findByText(/no public address configured/),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Redirect URI")).not.toBeInTheDocument()
  })

  it("saves a new connection with the fields entered", async () => {
    const requests = mockApi({ sso: organizationSso({ connection: null }) })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add connection" }),
    )
    await user.type(screen.getByLabelText("Display name"), "Acme Okta")
    await user.type(
      screen.getByLabelText("Issuer"),
      "https://login.acme.example",
    )
    await user.type(screen.getByLabelText("Client ID"), "acme-dashboard")
    // At least eight characters, which is what the gateway requires: a shorter
    // one is refused there, so a test using it proves nothing about a save.
    await user.type(screen.getByLabelText("Client secret"), "s3cr3t-98")
    await user.click(screen.getByRole("button", { name: "Add connection" }))

    await waitFor(() => {
      expect(
        requests.some(
          (request) => request.method === "PUT" && request.url.includes("/sso"),
        ),
      ).toBe(true)
    })
    const saved = requests.find((request) => request.method === "PUT")
    expect(saved?.body).toEqual({
      display_name: "Acme Okta",
      issuer: "https://login.acme.example",
      client_id: "acme-dashboard",
      client_secret: "s3cr3t-98",
      default_role: "member",
      enabled: true,
    })
    // The form closes only on a save the gateway accepted, and the page then
    // shows the stored connection, which is what separates a create from a
    // request that was merely sent.
    await waitFor(() => {
      expect(screen.queryByLabelText("Client secret")).not.toBeInTheDocument()
    })
    expect(await screen.findByText("Acme Okta")).toBeInTheDocument()
    expect(screen.getByText("••••t-98")).toBeInTheDocument()
  })

  it("keeps the stored secret when an edit leaves it blank", async () => {
    const requests = mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
    })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("button", { name: "Edit connection" }),
    )
    await user.clear(await screen.findByLabelText("Display name"))
    await user.type(screen.getByLabelText("Display name"), "Renamed IdP")
    await user.click(screen.getByRole("button", { name: "Save connection" }))

    await waitFor(() => {
      expect(requests.some((request) => request.method === "PUT")).toBe(true)
    })
    const saved = requests.find((request) => request.method === "PUT")
    expect(saved?.body).toMatchObject({
      client_secret: null,
      display_name: "Renamed IdP",
    })
  })

  it("asks for a new secret once the issuer changes", async () => {
    mockApi({ sso: organizationSso({ connection: oidcConnection() }) })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("button", { name: "Edit connection" }),
    )
    await user.clear(await screen.findByLabelText("Issuer"))
    await user.type(
      screen.getByLabelText("Issuer"),
      "https://elsewhere.example",
    )

    expect(
      screen.getByRole("button", { name: "Save connection" }),
    ).toBeDisabled()
  })

  it("re-reads the provider configuration for the saved connection", async () => {
    const requests = mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
    })
    renderPage(<OrganizationSsoPage />)

    await userEvent.click(
      await screen.findByRole("button", { name: "Refresh from provider" }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "POST" && request.url.includes("/sso/test"),
        ),
      ).toBe(true)
    })
  })

  it("removes a connection only through the confirm dialog", async () => {
    const requests = mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
    })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("button", { name: "Remove connection" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/Acme Okta is removed/)).toBeVisible()
    expect(requests.some((request) => request.method === "DELETE")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Remove connection" }),
    )

    await waitFor(() =>
      expect(requests.some((request) => request.method === "DELETE")).toBe(
        true,
      ),
    )
  })

  it("shows the preflight count and the list before enforcement can be enabled", async () => {
    mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
      preflight: organizationSsoPreflight({
        total_members: 3,
        affected_count: 2,
        affected_members: [
          organizationSsoPreflightMember({
            user_id: "11111111-aaaa-aaaa-aaaa-111111111111",
            full_name: "Alex Doe",
          }),
        ],
        affected_members_truncated: true,
      }),
    })
    renderPage(<OrganizationSsoPage />)

    expect(
      await screen.findByText(/3 members, 2 of whom sign in some other way/),
    ).toBeInTheDocument()
    expect(screen.getByText("Alex Doe")).toBeInTheDocument()
    expect(screen.getByText(/and 1 others/)).toBeInTheDocument()
    expect(
      screen.getByRole("checkbox", {
        name: "Require sign-in through this connection",
      }),
    ).not.toBeChecked()
  })

  it("says API keys stay on the old account and must be rotated", async () => {
    mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
      preflight: organizationSsoPreflight({
        total_members: 5,
        affected_count: 2,
        affected_members: [organizationSsoPreflightMember()],
      }),
    })
    renderPage(<OrganizationSsoPage />)

    expect(
      await screen.findByText(
        /Their API keys stay on the old account and cannot be moved, so they should issue new keys on the new account and revoke the old ones\./,
      ),
    ).toBeInTheDocument()
  })

  it("names the owner who keeps a non-SSO route", async () => {
    mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
      members: [
        organizationMember({
          user_id: "22222222-bbbb-bbbb-bbbb-222222222222",
          full_name: "Jane Owner",
          role: "owner",
          status: "active",
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("checkbox", {
        name: "Require sign-in through this connection",
      }),
    )
    await user.click(
      screen.getByRole("button", { name: /Owner who keeps a non-SSO route/ }),
    )
    await user.click(await screen.findByRole("option", { name: "Jane Owner" }))

    expect(
      await screen.findByText(/Jane Owner keeps a non-SSO route/),
    ).toBeInTheDocument()
  })

  it("states what the organization gives up when it waives the exception", async () => {
    mockApi({ sso: organizationSso({ connection: oidcConnection() }) })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("checkbox", {
        name: "Require sign-in through this connection",
      }),
    )
    await user.click(
      screen.getByRole("checkbox", {
        name: "This organization keeps no non-SSO route",
      }),
    )

    expect(
      await screen.findByText(
        /nobody in this organization can sign in until a deployment operator steps in/,
      ),
    ).toBeInTheDocument()
  })

  it("shows the connection's own refusal when nobody has signed in through it yet", async () => {
    mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
      enforcement: () =>
        jsonResponse(
          {
            detail:
              "Nobody has signed in through this organization's connection yet, so there is nothing proven to require.",
          },
          400,
        ),
    })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("checkbox", {
        name: "Require sign-in through this connection",
      }),
    )
    await user.click(screen.getByRole("button", { name: "Save enforcement" }))

    expect(
      await screen.findByText(
        "Nobody has signed in through this organization's connection yet, so there is nothing proven to require.",
      ),
    ).toBeInTheDocument()
  })

  it("shows its own refusal when neither an owner nor a waiver is named", async () => {
    mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
      enforcement: () =>
        jsonResponse(
          {
            detail:
              "Name one of this organization's owners to keep a non-SSO route, or waive that exception explicitly, before requiring sign-in through this connection.",
          },
          400,
        ),
    })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("checkbox", {
        name: "Require sign-in through this connection",
      }),
    )
    await user.click(screen.getByRole("button", { name: "Save enforcement" }))

    expect(
      await screen.findByText(
        "Name one of this organization's owners to keep a non-SSO route, or waive that exception explicitly, before requiring sign-in through this connection.",
      ),
    ).toBeInTheDocument()
  })

  it("shows its own refusal when the named account is not an owner", async () => {
    mockApi({
      sso: organizationSso({ connection: oidcConnection() }),
      members: [
        organizationMember({
          user_id: "33333333-cccc-cccc-cccc-333333333333",
          full_name: "Jane Owner",
          role: "owner",
          status: "active",
        }),
      ],
      enforcement: () =>
        jsonResponse(
          {
            detail:
              "Name one of this organization's owners to keep a non-SSO route, or waive that exception explicitly, before requiring sign-in through this connection.",
          },
          400,
        ),
    })
    const user = userEvent.setup()
    renderPage(<OrganizationSsoPage />)

    await user.click(
      await screen.findByRole("checkbox", {
        name: "Require sign-in through this connection",
      }),
    )
    await user.click(
      screen.getByRole("button", { name: /Owner who keeps a non-SSO route/ }),
    )
    await user.click(await screen.findByRole("option", { name: "Jane Owner" }))
    await user.click(screen.getByRole("button", { name: "Save enforcement" }))

    expect(
      await screen.findByText(
        "Name one of this organization's owners to keep a non-SSO route, or waive that exception explicitly, before requiring sign-in through this connection.",
      ),
    ).toBeInTheDocument()
  })
})
