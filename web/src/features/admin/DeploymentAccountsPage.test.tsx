import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentUser, PasswordLink } from "@/client"
import { DeploymentAccountsPage } from "@/features/admin/DeploymentAccountsPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, deploymentUser } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

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
  granted?: boolean
  accounts?: DeploymentUser[]
  count?: number
  passwordLink?: PasswordLink
}) {
  const granted = opts.granted ?? true
  const accounts = opts.accounts ?? [deploymentUser()]
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes(`${API_ROOT}/admin/access`)) {
      return jsonResponse({ granted })
    }
    if (url.endsWith("/password-link")) {
      return jsonResponse(opts.passwordLink, 201)
    }
    if (url.includes(`${API_ROOT}/admin/users`)) {
      if (method === "GET") {
        return jsonResponse({
          data: accounts,
          count: opts.count ?? accounts.length,
        })
      }
      const body = init?.body ? JSON.parse(String(init.body)) : {}
      return jsonResponse({ ...accounts[0], ...body })
    }
    return jsonResponse({})
  })

  return requests
}

function renderPage(ui: ReactElement, url = "/admin/accounts") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
    { url },
  )
}

function rowFor(name: string) {
  return screen
    .getAllByRole("row")
    .find((row) => within(row).queryByText(name) !== null) as HTMLElement
}

const ANALYST = deploymentUser({
  id: "bbbbbbbb-0000-0000-0000-000000000000",
  full_name: "Analyst",
  email: "analyst@example.com",
})

const OPERATOR = deploymentUser({
  id: "aaaaaaaa-0000-0000-0000-000000000000",
  full_name: "Operator",
  email: null,
  is_superuser: true,
  is_bootstrap_operator: true,
  is_self: true,
  last_sign_in_at: "2026-01-01T00:00:00+00:00",
})

const realMatchMedia = window.matchMedia

function stubPhone() {
  window.matchMedia = ((query: string) => ({
    matches: true,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  })) as typeof window.matchMedia
}

afterEach(() => {
  vi.restoreAllMocks()
  window.matchMedia = realMatchMedia
})

describe("DeploymentAccountsPage", () => {
  it("lists every account with its organizations, sign-in and standing", async () => {
    mockApi({ accounts: [OPERATOR, ANALYST] })
    await renderPage(<DeploymentAccountsPage />)

    expect(await screen.findByText("Analyst")).toBeInTheDocument()
    expect(screen.getByText("analyst@example.com")).toBeInTheDocument()
    const analyst = rowFor("Analyst")
    expect(
      within(analyst).getByText("Default organization (member)"),
    ).toBeInTheDocument()
    // Never signed in reads as "never" rather than a dash: the page is about
    // whether the account is still in use, and no stamp is the finding.
    expect(within(analyst).getByText("never")).toBeInTheDocument()
    expect(
      within(rowFor("Operator")).getByText("BOOTSTRAP OPERATOR"),
    ).toBeInTheDocument()
  })

  it("keeps an account's actions on its card on a phone", async () => {
    stubPhone()
    mockApi({ accounts: [OPERATOR, ANALYST] })
    await renderPage(<DeploymentAccountsPage />)

    expect(await screen.findByText("Analyst")).toBeInTheDocument()
    expect(screen.queryByRole("grid")).not.toBeInTheDocument()
    const card = screen.getByText("Analyst").closest("li") as HTMLElement
    expect(within(card).getAllByRole("button").length).toBeGreaterThan(0)
  })

  it("marks a membership the organization roster would have dropped", async () => {
    mockApi({
      accounts: [
        deploymentUser({
          full_name: "Stuck",
          organizations: [
            {
              organization_id: "11111111-1111-1111-1111-111111111111",
              name: "Default organization",
              slug: "default",
              role: "member",
              status: "suspended",
            },
          ],
        }),
      ],
    })
    await renderPage(<DeploymentAccountsPage />)

    expect(
      await screen.findByText("Default organization (member, suspended)"),
    ).toBeInTheDocument()
  })

  it("deactivates an account after a confirmation", async () => {
    const requests = mockApi({ accounts: [ANALYST] })
    await renderPage(<DeploymentAccountsPage />)

    await userEvent.click(await screen.findByLabelText("Deactivate Analyst"))
    await userEvent.click(
      screen.getByRole("button", { name: "Deactivate account" }),
    )

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toContain(`${API_ROOT}/admin/users/${ANALYST.id}`)
    expect(patch?.body).toEqual({ is_active: false })
  })

  it("hands over a password link to share where mail did not go out", async () => {
    const requests = mockApi({
      accounts: [ANALYST],
      passwordLink: {
        mail_sent: false,
        link: "/#/reset-password?token=shared.abc",
        expires_at: "2026-10-14T12:00:00Z",
      },
    })
    await renderPage(<DeploymentAccountsPage />)

    await userEvent.click(
      await screen.findByLabelText("Send Analyst a password link"),
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Create the link" }),
    )

    const post = requests.find((request) => request.method === "POST")
    expect(post?.url).toContain(
      `${API_ROOT}/admin/users/${ANALYST.id}/password-link`,
    )
    expect(
      await screen.findByText(/Otari did not send the email/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        `${window.location.origin}${window.location.pathname}#/reset-password?token=shared.abc`,
      ),
    ).toBeInTheDocument()
  })

  it("says the link was emailed, and shows none, where mail went out", async () => {
    mockApi({
      accounts: [ANALYST],
      passwordLink: {
        mail_sent: true,
        link: null,
        expires_at: "2026-10-14T12:00:00Z",
      },
    })
    await renderPage(<DeploymentAccountsPage />)

    await userEvent.click(
      await screen.findByLabelText("Send Analyst a password link"),
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Create the link" }),
    )

    expect(
      await screen.findByText(/link to set a new password was sent to/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/reset-password/)).not.toBeInTheDocument()
  })

  it("offers no password link for a deactivated account, and says why", async () => {
    mockApi({
      accounts: [deploymentUser({ full_name: "Analyst", is_active: false })],
    })
    await renderPage(<DeploymentAccountsPage />)

    expect(
      await screen.findByLabelText(
        "Send Analyst a password link (Reactivate this account first)",
      ),
    ).toBeDisabled()
  })

  it("reactivates a deactivated account with no confirmation", async () => {
    const requests = mockApi({
      accounts: [deploymentUser({ full_name: "Analyst", is_active: false })],
    })
    await renderPage(<DeploymentAccountsPage />)

    await userEvent.click(await screen.findByLabelText("Reactivate Analyst"))

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.body).toEqual({ is_active: true })
  })

  it("grants operator access on its own, leaving the active flag alone", async () => {
    const requests = mockApi({ accounts: [ANALYST] })
    await renderPage(<DeploymentAccountsPage />)

    await userEvent.click(
      await screen.findByLabelText("Grant operator access to Analyst"),
    )

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.body).toEqual({ is_superuser: true })
  })

  it("disables the two lockout controls on the caller's own row, with the reason", async () => {
    // A caller who is not the bootstrap identity, which is what makes the *self*
    // reason the one on the control: a row that is both takes the bootstrap
    // reason, because the self reason's remedy ("another operator") is one no
    // operator can carry out on the bootstrap row. `accounts.test.ts` pins that
    // order; this pins the plain case.
    const SELF = deploymentUser({
      id: "cccccccc-0000-0000-0000-000000000000",
      full_name: "Reader",
      is_superuser: true,
      is_self: true,
    })
    mockApi({ accounts: [SELF, ANALYST] })
    await renderPage(<DeploymentAccountsPage />)

    const own = await screen.findByLabelText(
      /^Deactivate Reader \(This is your own account/,
    )
    expect(own).toBeDisabled()
    expect(
      screen.getByLabelText(
        /^Remove operator access from Reader \(This is your own account/,
      ),
    ).toBeDisabled()
    // The other row keeps both, so the guard is about who the row is and not
    // about the page being read-only.
    expect(screen.getByLabelText("Deactivate Analyst")).toBeEnabled()
  })

  it("disables them on the caller's own row when it is also the bootstrap one", async () => {
    // The ordinary standalone shape: one operator identity, marked, and reading
    // its own row. Both controls are still off, and the reason is the one that
    // stays true whoever is reading.
    mockApi({ accounts: [OPERATOR, ANALYST] })
    await renderPage(<DeploymentAccountsPage />)

    expect(
      await screen.findByLabelText(
        /^Deactivate Operator \(The bootstrap operator/,
      ),
    ).toBeDisabled()
    expect(
      screen.getByLabelText(
        /^Remove operator access from Operator \(The bootstrap operator/,
      ),
    ).toBeDisabled()
  })

  it("disables them on the bootstrap operator seen by another operator", async () => {
    mockApi({
      accounts: [
        deploymentUser({
          full_name: "Anchor",
          is_superuser: true,
          is_bootstrap_operator: true,
        }),
      ],
    })
    await renderPage(<DeploymentAccountsPage />)

    expect(
      await screen.findByLabelText(
        /^Deactivate Anchor \(The bootstrap operator/,
      ),
    ).toBeDisabled()
  })

  it("asks the server for the search term, not the page on screen", async () => {
    const requests = mockApi({ accounts: [OPERATOR, ANALYST] })
    await renderPage(<DeploymentAccountsPage />)
    await screen.findByText("Analyst")

    await userEvent.type(
      screen.getByRole("searchbox", { name: "Search accounts" }),
      "analyst@",
    )

    await waitFor(() => {
      const searched = requests.filter((request) =>
        request.url.includes("search=analyst%40"),
      )
      expect(searched).not.toHaveLength(0)
    })
    // Debounced: one request for the settled term, not one per keystroke.
    expect(
      requests.filter((request) => request.url.includes("search=")),
    ).toHaveLength(1)
  })

  it("pages through the accounts on the server", async () => {
    const requests = mockApi({ accounts: [OPERATOR, ANALYST], count: 60 })
    await renderPage(<DeploymentAccountsPage />, "/admin/accounts?page=1")
    await screen.findByText("Analyst")

    expect(
      requests.some((request) =>
        request.url.includes("/admin/users?skip=25&limit=25"),
      ),
    ).toBe(true)
  })

  it("falls back to a valid page and size when the URL carries neither", async () => {
    const requests = mockApi({ accounts: [OPERATOR, ANALYST] })
    await renderPage(
      <DeploymentAccountsPage />,
      "/admin/accounts?page=-3&size=0",
    )
    await screen.findByText("Analyst")

    const listed = requests.filter(
      (request) =>
        request.method === "GET" && request.url.includes("/admin/users?"),
    )
    expect(listed.length).toBeGreaterThan(0)
    for (const request of listed) {
      expect(request.url).toContain("skip=0&limit=25")
    }
  })

  it("says the surface is not for you when the gate refuses", async () => {
    mockApi({ granted: false })
    await renderPage(<DeploymentAccountsPage />)

    expect(
      await screen.findByText("Accounts is not available to you"),
    ).toBeInTheDocument()
    expect(screen.queryByRole("grid")).not.toBeInTheDocument()
  })

  it("reports a gate that failed rather than calling the page not yours", async () => {
    // A 500 leaves `granted` false the same way a refusal does, and the two are
    // not the same thing to tell an operator.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).includes(`${API_ROOT}/admin/access`)
        ? jsonResponse({ detail: "Database error" }, 500)
        : jsonResponse({}),
    )
    await renderPage(<DeploymentAccountsPage />)

    expect(await screen.findByText("Database error")).toBeInTheDocument()
    expect(screen.queryByText("Accounts is not available to you")).toBeNull()
  })

  it("does not fetch the list before the gate has answered yes", async () => {
    const requests = mockApi({ granted: false })
    await renderPage(<DeploymentAccountsPage />)

    await screen.findByText("Accounts is not available to you")
    expect(
      requests.some((request) =>
        request.url.includes(`${API_ROOT}/admin/users`),
      ),
    ).toBe(false)
  })
})
