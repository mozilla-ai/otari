import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext } from "@/client"
import { OrganizationGeneralPage } from "@/features/organization/OrganizationGeneralPage"
import { organizationContext } from "@/tests/fixtures"

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

function mockApi(context: OrganizationContext = organizationContext()) {
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    requests.push({
      url: String(input),
      method: (init?.method ?? "GET").toUpperCase(),
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    return jsonResponse(context)
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

describe("OrganizationGeneralPage", () => {
  it("names the organization the caller is pointed at and their role in it", async () => {
    mockApi()
    renderPage(<OrganizationGeneralPage />)

    expect(
      await screen.findByDisplayValue("Default Organization"),
    ).toBeInTheDocument()
    expect(screen.getByText(/Your role here is owner/)).toBeInTheDocument()
    expect(screen.getByText("default-organization")).toBeInTheDocument()
  })

  it("renames the organization through the active-organization endpoint", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    const name = await screen.findByDisplayValue("Default Organization")
    await user.clear(name)
    await user.type(name, "Platform")
    await user.click(screen.getByRole("button", { name: "Save name" }))

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toContain("/v1/organizations/me")
    expect(patch?.body).toEqual({ name: "Platform" })
  })

  it("will not save a name that has not changed", async () => {
    mockApi()
    renderPage(<OrganizationGeneralPage />)

    await screen.findByDisplayValue("Default Organization")
    expect(screen.getByRole("button", { name: "Save name" })).toBeDisabled()
  })

  it("leaves creating and switching to the scope switcher, and offers no delete", async () => {
    mockApi()
    renderPage(<OrganizationGeneralPage />)

    await screen.findByDisplayValue("Default Organization")
    // Both controls exist, in the switcher above the rail: they are about which
    // organization you are looking at, where this page is about the one you are
    // in. A second copy here would be a second thing to keep in step.
    expect(
      screen.queryByRole("button", { name: /Create organization/ }),
    ).toBeNull()
    expect(screen.queryByRole("button", { name: "Switch" })).toBeNull()
    // Delete is the one with no endpoint anywhere, so a control for it would be
    // a 404 waiting to happen.
    expect(
      screen.queryByRole("button", { name: /Delete organization/ }),
    ).toBeNull()
    expect(screen.queryByText("Danger zone")).toBeNull()
  })

  it("leaves the name read-only for a caller who cannot manage the organization", async () => {
    mockApi(organizationContext({ role: "member" }))
    renderPage(<OrganizationGeneralPage />)

    await screen.findByDisplayValue("Default Organization")
    expect(screen.getByRole("button", { name: "Save name" })).toBeDisabled()
    expect(
      screen.getByText(/Only owners and admins can change it/),
    ).toBeInTheDocument()
  })

  it("reports a context that could not be read instead of an empty page", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Tenancy is unavailable" }, 500),
    )
    renderPage(<OrganizationGeneralPage />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tenancy is unavailable",
    )
  })
})
