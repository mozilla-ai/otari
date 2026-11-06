import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, OrganizationSso } from "@/client"
import { OrganizationSsoPage } from "@/features/organization/OrganizationSsoPage"
import { formatDate } from "@/shared/helpers/format"
import {
  oidcConnection,
  oidcConnectionDomain,
  organizationContext,
  organizationSso,
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
  /** What a save answers, so a test can drive the refusal path. */
  save?: () => Response
}

function mockApi(opts: MockOpts = {}) {
  const sso = opts.sso ?? organizationSso()
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
      return jsonResponse(sso)
    }
    if (url.includes("/sso")) {
      if (method === "GET") return jsonResponse(sso)
      if (method === "PUT") {
        return opts.save ? opts.save() : jsonResponse(sso)
      }
      if (method === "DELETE") return jsonResponse(undefined, 204)
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
    expect(screen.getByText(/stops admitting anyone/)).toBeInTheDocument()
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
    await user.type(screen.getByLabelText("Client secret"), "s3cr3t")
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
      client_secret: "s3cr3t",
      default_role: "member",
      enabled: true,
    })
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
})
