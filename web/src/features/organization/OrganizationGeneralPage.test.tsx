import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext } from "@/client"
import { OrganizationGeneralPage } from "@/features/organization/OrganizationGeneralPage"
import { organization, organizationContext } from "@/tests/fixtures"

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

function mockApi(opts: {
  context?: OrganizationContext
  memberships?: OrganizationContext[]
}) {
  const context = opts.context ?? organizationContext()
  const memberships = opts.memberships ?? [context]
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/v1/organizations/me/memberships")) {
      return jsonResponse({ data: memberships, count: memberships.length })
    }
    if (url.includes("/v1/organizations/me/switch")) {
      return jsonResponse(context)
    }
    if (url.includes("/v1/organizations/me")) {
      return jsonResponse(context)
    }
    return jsonResponse({})
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
    mockApi({})
    renderPage(<OrganizationGeneralPage />)

    expect(
      await screen.findByDisplayValue("Default Organization"),
    ).toBeInTheDocument()
    expect(screen.getByText(/Your role here is owner/)).toBeInTheDocument()
    expect(screen.getByText("default-organization")).toBeInTheDocument()
  })

  it("renames the organization through the active-organization endpoint", async () => {
    const requests = mockApi({})
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
    mockApi({})
    renderPage(<OrganizationGeneralPage />)

    await screen.findByDisplayValue("Default Organization")
    expect(screen.getByRole("button", { name: "Save name" })).toBeDisabled()
  })

  it("offers no switcher and refuses deletion while there is one organization", async () => {
    mockApi({})
    renderPage(<OrganizationGeneralPage />)

    await screen.findByDisplayValue("Default Organization")
    expect(screen.queryByRole("button", { name: "Switch" })).toBeNull()
    // Every identity has to be pointed at an organization, so the server
    // refuses this; saying why beats reporting a 400 after the click.
    expect(
      screen.getByRole("button", { name: "Delete organization" }),
    ).toBeDisabled()
    expect(screen.getByText(/only organization/)).toBeInTheDocument()
  })

  it("switches into another organization the caller belongs to", async () => {
    const other = organizationContext({
      organization_member_id: "other-membership",
      organization: organization({
        id: "99999999-9999-9999-9999-999999999999",
        name: "Research",
        slug: "research",
      }),
    })
    const requests = mockApi({
      memberships: [organizationContext(), other],
    })
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    const picker = await screen.findByLabelText("Organization")
    await user.selectOptions(picker, "99999999-9999-9999-9999-999999999999")
    await user.click(screen.getByRole("button", { name: "Switch" }))

    const post = requests.find((request) =>
      request.url.includes("/v1/organizations/me/switch"),
    )
    expect(post?.body).toEqual({
      organization_id: "99999999-9999-9999-9999-999999999999",
    })
  })

  it("hides the destructive and editing controls from a non-manager", async () => {
    mockApi({ context: organizationContext({ role: "member" }) })
    renderPage(<OrganizationGeneralPage />)

    await screen.findByDisplayValue("Default Organization")
    expect(screen.getByRole("button", { name: "Save name" })).toBeDisabled()
    // Deletion is an owner's alone, so the whole danger zone is absent rather
    // than present and refused.
    expect(screen.queryByText("Danger zone")).toBeNull()
    expect(
      screen.getByText(/Only owners and admins can change it/),
    ).toBeInTheDocument()
  })

  it("reports a bootstrap that could not be read instead of an empty page", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Tenancy is unavailable" }, 500),
    )
    renderPage(<OrganizationGeneralPage />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tenancy is unavailable",
    )
  })
})
