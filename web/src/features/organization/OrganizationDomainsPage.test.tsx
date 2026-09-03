import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, OrganizationDomain } from "@/client"
import { OrganizationDomainsPage } from "@/features/organization/OrganizationDomainsPage"
import { organizationContext, organizationDomain } from "@/tests/fixtures"

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
  domains?: OrganizationDomain[]
  context?: OrganizationContext
  /** What the verify call answers, so a test can drive the refusal path. */
  verify?: () => Response
}

function mockApi(opts: MockOpts = {}) {
  const domains = opts.domains ?? []
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/verify")) {
      return opts.verify ? opts.verify() : jsonResponse(domains[0])
    }
    if (url.includes("/domains")) {
      if (method === "GET") {
        return jsonResponse({ count: domains.length, data: domains })
      }
      // Every write is re-read through the invalidated list, so one row serves.
      return jsonResponse(domains[0] ?? organizationDomain())
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

describe("OrganizationDomainsPage", () => {
  it("shows an unverified claim as inert, with the record to publish", async () => {
    mockApi({ domains: [organizationDomain()] })
    renderPage(<OrganizationDomainsPage />)

    expect(
      await screen.findByRole("rowheader", { name: "acme.example" }),
    ).toBeInTheDocument()
    expect(screen.getByText("Not verified")).toBeInTheDocument()
    // The whole string to paste, not the bare token: an admin has no use for
    // the token on its own and prepending the prefix is what gets fumbled.
    expect(screen.getByLabelText("TXT record for acme.example")).toHaveValue(
      "otari-domain-verification=tok-abc123",
    )
  })

  it("offers no pause control until the claim is proven", async () => {
    // Pausing an unverified claim would imply it was admitting people.
    mockApi({ domains: [organizationDomain()] })
    renderPage(<OrganizationDomainsPage />)

    expect(
      await screen.findByRole("rowheader", { name: "acme.example" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Pause" })).toBeNull()
    expect(screen.getByRole("button", { name: "Remove" })).toBeInTheDocument()
  })

  it("shows a verified, enabled claim as active and pausable", async () => {
    mockApi({
      domains: [
        organizationDomain({ verified_at: "2026-08-25T00:00:00+00:00" }),
      ],
    })
    renderPage(<OrganizationDomainsPage />)

    expect(await screen.findByText("Active")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument()
    // The proof panel is retired once there is nothing left to publish.
    expect(screen.queryByText(/Publish this as a TXT record/)).toBeNull()
  })

  it("distinguishes a paused claim from an unverified one", async () => {
    mockApi({
      domains: [
        organizationDomain({
          verified_at: "2026-08-25T00:00:00+00:00",
          enabled: false,
        }),
      ],
    })
    renderPage(<OrganizationDomainsPage />)

    expect(await screen.findByText("Paused")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument()
  })

  it("claims a domain at the role the form names", async () => {
    const requests = mockApi()
    renderPage(<OrganizationDomainsPage />)

    await userEvent.click(
      await screen.findByRole("button", { name: "Claim domain" }),
    )
    await userEvent.type(screen.getByLabelText(/Domain/), "acme.example")
    await userEvent.click(screen.getByRole("button", { name: "Claim domain" }))

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "POST" && request.url.includes("/domains"),
        ),
      ).toBe(true)
    })
    const created = requests.find((request) => request.method === "POST")
    expect(created?.body).toEqual({
      domain: "acme.example",
      default_role: "member",
      enabled: true,
    })
  })

  it("never offers a management role, because a DNS record must not mint admins", async () => {
    mockApi()
    renderPage(<OrganizationDomainsPage />)

    await userEvent.click(
      await screen.findByRole("button", { name: "Claim domain" }),
    )
    const picker = screen.getByRole("button", { name: /They join as/ })
    await userEvent.click(picker)

    expect(await screen.findByRole("option", { name: "Member" })).toBeVisible()
    expect(screen.getByRole("option", { name: "Viewer" })).toBeVisible()
    expect(screen.queryByRole("option", { name: "Owner" })).toBeNull()
    expect(screen.queryByRole("option", { name: "Admin" })).toBeNull()
  })

  it("sends a verify request for the pending claim", async () => {
    const requests = mockApi({ domains: [organizationDomain()] })
    renderPage(<OrganizationDomainsPage />)

    await userEvent.click(
      await screen.findByRole("button", { name: "Verify domain" }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "POST" && request.url.includes("/verify"),
        ),
      ).toBe(true)
    })
  })

  it("surfaces the refusal when the record is not published yet", async () => {
    // The expected first answer after publishing, so it has to read as a
    // normal next step rather than as a broken page.
    mockApi({
      domains: [organizationDomain()],
      verify: () =>
        jsonResponse(
          { detail: "No matching TXT record was found at acme.example." },
          400,
        ),
    })
    renderPage(<OrganizationDomainsPage />)

    await userEvent.click(
      await screen.findByRole("button", { name: "Verify domain" }),
    )

    expect(
      await screen.findByText(/No matching TXT record/),
    ).toBeInTheDocument()
  })

  it("tells a plain member they cannot manage domains, and reads nothing", async () => {
    const requests = mockApi({
      context: organizationContext({ role: "member" }),
    })
    renderPage(<OrganizationDomainsPage />)

    expect(
      await screen.findByText(
        "Only organization owners and admins can manage email domains.",
      ),
    ).toBeInTheDocument()
    // Withheld rather than fired and refused: the banner is the whole answer.
    expect(requests.some((request) => request.url.includes("/domains"))).toBe(
      false,
    )
  })
})
