import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext } from "@/client"
import { OrganizationGeneralPage } from "@/features/organization/OrganizationGeneralPage"
import { API_ROOT } from "@/shared/api/client"
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

// Opens the rename dialog and hands back its scope, so an assertion about the
// dialog cannot be satisfied by the page underneath it (both show the name).
async function openRenameDialog(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    await screen.findByRole("button", { name: "Change organization name" }),
  )
  return within(await screen.findByRole("dialog"))
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationGeneralPage", () => {
  it("names the organization the caller is pointed at and their role in it", async () => {
    mockApi()
    renderPage(<OrganizationGeneralPage />)

    expect(await screen.findByText("Default Organization")).toBeInTheDocument()
    expect(screen.getByText(/Your role here is owner/)).toBeInTheDocument()
    expect(screen.getByText("default-organization")).toBeInTheDocument()
  })

  it("keeps the name out of an editable box until the rename is asked for", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    // The accident this page is guarded against: a focusable input holding the
    // tenant's name, sitting open on arrival.
    await screen.findByText("Default Organization")
    expect(screen.queryByRole("textbox")).toBeNull()

    const dialog = await openRenameDialog(user)
    expect(dialog.getByRole("textbox", { name: /New name/ })).toHaveValue(
      "Default Organization",
    )
  })

  it("shows the name being replaced beside the one being typed", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    const dialog = await openRenameDialog(user)
    await user.clear(dialog.getByRole("textbox", { name: /New name/ }))
    await user.type(
      dialog.getByRole("textbox", { name: /New name/ }),
      "Platform",
    )

    expect(dialog.getByText("Current name")).toBeInTheDocument()
    expect(dialog.getByText("Default Organization")).toBeInTheDocument()
  })

  it("renames the organization through the active-organization endpoint", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    const dialog = await openRenameDialog(user)
    const name = dialog.getByRole("textbox", { name: /New name/ })
    await user.clear(name)
    await user.type(name, "Platform")
    await user.click(
      dialog.getByRole("button", { name: "Change organization name" }),
    )

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toContain(`${API_ROOT}/organizations/me`)
    expect(patch?.body).toEqual({ name: "Platform" })
  })

  it("closes the dialog once the rename lands", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    const dialog = await openRenameDialog(user)
    const name = dialog.getByRole("textbox", { name: /New name/ })
    await user.clear(name)
    await user.type(name, "Platform")
    await user.click(
      dialog.getByRole("button", { name: "Change organization name" }),
    )

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
  })

  it("will not save a name that has not changed", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    const dialog = await openRenameDialog(user)
    expect(
      dialog.getByRole("button", { name: "Change organization name" }),
    ).toBeDisabled()
  })

  it("reports a rejected rename in the dialog rather than closing over it", async () => {
    const user = userEvent.setup()
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      if ((init?.method ?? "GET").toUpperCase() === "PATCH") {
        return jsonResponse({ detail: "Name already taken" }, 409)
      }
      return jsonResponse(organizationContext())
    })
    renderPage(<OrganizationGeneralPage />)

    const dialog = await openRenameDialog(user)
    const name = dialog.getByRole("textbox", { name: /New name/ })
    await user.clear(name)
    await user.type(name, "Platform")
    await user.click(
      dialog.getByRole("button", { name: "Change organization name" }),
    )

    expect(await dialog.findByRole("alert")).toHaveTextContent(
      "Name already taken",
    )
  })

  it("leaves creating and switching to the scope switcher", async () => {
    mockApi()
    renderPage(<OrganizationGeneralPage />)

    await screen.findByText("Default Organization")
    // Both controls exist, in the switcher above the rail: they are about which
    // organization you are looking at, where this page is about the one you are
    // in. A second copy here would be a second thing to keep in step.
    expect(
      screen.queryByRole("button", { name: /Create organization/ }),
    ).toBeNull()
    expect(screen.queryByRole("button", { name: "Switch" })).toBeNull()
  })

  it("deletes the organization through the active-organization endpoint once confirmed", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationGeneralPage />)

    await user.click(
      await screen.findByRole("button", { name: "Delete organization" }),
    )
    const dialog = within(await screen.findByRole("alertdialog"))
    expect(dialog.getByText(/Default Organization is deleted/)).toBeVisible()
    await user.click(
      dialog.getByRole("button", { name: "Delete organization" }),
    )

    const deleted = requests.find((request) => request.method === "DELETE")
    expect(deleted?.url).toBe(`${API_ROOT}/organizations/me`)
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull())
  })

  it("reports a refused delete in the dialog", async () => {
    const user = userEvent.setup()
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      if ((init?.method ?? "GET").toUpperCase() === "DELETE") {
        return jsonResponse(
          { detail: "Only an organization with no usage can be deleted." },
          409,
        )
      }
      return jsonResponse(organizationContext())
    })
    renderPage(<OrganizationGeneralPage />)

    await user.click(
      await screen.findByRole("button", { name: "Delete organization" }),
    )
    const dialog = within(await screen.findByRole("alertdialog"))
    await user.click(
      dialog.getByRole("button", { name: "Delete organization" }),
    )

    expect(await dialog.findByRole("alert")).toHaveTextContent(
      "Only an organization with no usage can be deleted.",
    )
  })

  it("offers no delete for the deployment's default organization", async () => {
    // The gateway refuses it, so a control for it would only lead to a refusal.
    mockApi(
      organizationContext({ organization: organization({ slug: "default" }) }),
    )
    renderPage(<OrganizationGeneralPage />)

    await screen.findByText("Default Organization")
    expect(
      screen.queryByRole("button", { name: "Delete organization" }),
    ).toBeNull()
  })

  it("offers no rename to a caller who cannot manage the organization", async () => {
    mockApi(organizationContext({ role: "member" }))
    renderPage(<OrganizationGeneralPage />)

    expect(
      await screen.findByText(/Only owners and admins can change it/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Change organization name" }),
    ).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Delete organization" }),
    ).toBeNull()
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
