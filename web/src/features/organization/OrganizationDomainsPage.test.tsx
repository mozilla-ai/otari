import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
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
    expect(
      screen.getByRole("button", { name: "Remove domain" }),
    ).toBeInTheDocument()
  })

  it("removes a claim only through the confirm dialog", async () => {
    // otari-ai#2110. The dialog names the domain and says what survives the
    // removal, which the two-click button had no room to.
    const requests = mockApi({ domains: [organizationDomain()] })
    const user = userEvent.setup()
    renderPage(<OrganizationDomainsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Remove domain" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/acme.example stops admitting anyone/),
    ).toBeVisible()
    expect(requests.some((request) => request.method === "DELETE")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Remove claim" }),
    )

    await waitFor(() =>
      expect(requests.some((request) => request.method === "DELETE")).toBe(
        true,
      ),
    )
  })

  it("shows a verified, enabled claim as active and pausable", async () => {
    mockApi({
      domains: [
        organizationDomain({
          verified_at: "2026-08-25T00:00:00+00:00",
          proof_expires_at: "2099-01-01T00:00:00+00:00",
        }),
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
          proof_expires_at: "2099-01-01T00:00:00+00:00",
          enabled: false,
        }),
      ],
    })
    renderPage(<OrganizationDomainsPage />)

    expect(await screen.findByText("Paused")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument()
  })

  it("shows a claim whose proof has aged out as expired, not as unverified", async () => {
    // The claim *was* proven, so it is one re-check from working. Calling that
    // "Not verified" would suggest the admin never published the record.
    mockApi({
      domains: [
        organizationDomain({
          verified_at: "2026-05-01T00:00:00+00:00",
          proof_expires_at: "2026-07-30T00:00:00+00:00",
        }),
      ],
    })
    renderPage(<OrganizationDomainsPage />)

    expect(await screen.findByText("Proof expired")).toBeInTheDocument()
    expect(screen.queryByText("Active")).toBeNull()
    // And the card comes back, so there is somewhere to act.
    expect(
      screen.getByRole("button", { name: "Re-verify domain" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/proof has expired/)).toBeInTheDocument()
  })

  it("leaves a claim whose proof is still current alone", async () => {
    mockApi({
      domains: [
        organizationDomain({
          verified_at: "2026-08-25T00:00:00+00:00",
          proof_expires_at: "2099-01-01T00:00:00+00:00",
        }),
      ],
    })
    renderPage(<OrganizationDomainsPage />)

    expect(await screen.findByText("Active")).toBeInTheDocument()
    expect(screen.queryByText("Proof expired")).toBeNull()
    expect(screen.queryByRole("button", { name: /verify/i })).toBeNull()
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

  it("keeps the header trigger on screen while the dialog is open", async () => {
    // The dialog sits over the page rather than replacing the action, so the
    // control that opened it does not vanish from under the pointer.
    mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationDomainsPage />)

    const trigger = await screen.findByRole("button", { name: "Claim domain" })
    await user.click(trigger)

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    expect(trigger).toBeVisible()
  })

  it("opens on a blank draft after a close, not on the last one typed", async () => {
    // Reset on the way in: clearing on the way out would blank the fields
    // while the dialog is still animating away.
    mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationDomainsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Claim domain" }),
    )
    await user.type(screen.getByLabelText(/Domain/), "acme.example")
    // A draft this far along is dirty, so the way out is through the guard.
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

    await user.click(screen.getByRole("button", { name: "Claim domain" }))
    expect(await screen.findByLabelText(/Domain/)).toHaveValue("")
  })

  it("guards a changed role on the way out, with nothing typed", async () => {
    // The role is the other thing this form owns, and it is a choice the
    // operator cannot retype: without it in `isDirty`, closing discards it
    // silently. Dismissed through Cancel rather than Escape because in jsdom
    // focus lands on `<body>` after picking from the Select, so a keystroke
    // reaches nothing.
    mockApi()
    const user = userEvent.setup()
    renderPage(<OrganizationDomainsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Claim domain" }),
    )
    await user.click(screen.getByRole("button", { name: /They join as/ }))
    await user.click(await screen.findByRole("option", { name: "Viewer" }))

    await user.click(screen.getByRole("button", { name: "Cancel" }))

    expect(
      await screen.findByRole("button", { name: "Discard" }),
    ).toBeInTheDocument()
    // Still open behind the guard, so Keep editing returns to the choice
    // rather than to an empty form.
    await user.click(screen.getByRole("button", { name: "Keep editing" }))
    expect(
      screen.getByRole("button", { name: /They join as/ }),
    ).toHaveTextContent("Viewer")

    // And the seed is per open, not per page: the ref lives below the key, so
    // the next open starts clean. Held above it, or with the key dropped, the
    // dialog would arrive already dirty and Escape would ask before closing an
    // untouched form.
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    await user.click(screen.getByRole("button", { name: "Claim domain" }))
    await screen.findByRole("dialog")
    await user.keyboard("{Escape}")
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
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
