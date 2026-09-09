import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { PendingInvitationsPage } from "@/features/invitations/PendingInvitationsPage"
import { ApiError, apiFetch } from "@/shared/api/client"
import { pendingOrganizationInvitation } from "@/tests/fixtures"

// The network boundary, not the hooks: the real query keys, paths and
// invalidation run, so an empty state or a refusal comes from the hook logic
// rather than a stub of it.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

const INBOX = "/v1/organizations/me/pending-memberships"

interface Recorded {
  url: string
  method: string
}

function mockApi(
  options: {
    waiting?: ReturnType<typeof pendingOrganizationInvitation>[]
    listError?: number
    writeError?: string
  } = {},
) {
  const requests: Recorded[] = []
  // Re-read after a write, so the list a successful accept or decline
  // invalidates comes back without the row rather than being asserted from the
  // same fixture the first render used.
  let waiting = options.waiting ?? [pendingOrganizationInvitation()]
  vi.mocked(apiFetch).mockImplementation(async (path, init) => {
    const url = String(path)
    const method = init?.method ?? "GET"
    requests.push({ url, method })
    if (url.endsWith("/accept") || url.endsWith("/decline")) {
      if (options.writeError) {
        throw new ApiError(404, options.writeError)
      }
      waiting = []
      return (
        url.endsWith("/accept")
          ? { organization_name: "Research", role: "member" }
          : { message: "Invitation declined" }
      ) as never
    }
    if (url.startsWith(INBOX)) {
      if (options.listError) {
        throw new ApiError(options.listError, "Not found")
      }
      return { data: waiting, count: waiting.length } as never
    }
    return {} as never
  })
  return requests
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <PendingInvitationsPage />
    </QueryClientProvider>,
  )
}

describe("the invitee's membership inbox", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("names the organization, the role on offer, and the address it reached", async () => {
    mockApi()
    renderPage()

    expect(await screen.findByText("Research")).toBeVisible()
    expect(
      await screen.findByText(/Invited as Member to invitee@example.com/),
    ).toBeVisible()
    expect(await screen.findByText(/^Expires /)).toBeVisible()
  })

  it("accepts by membership id, never by token", async () => {
    const requests = mockApi()
    renderPage()

    const user = userEvent.setup()
    await user.click(
      await screen.findByRole("button", {
        name: "Accept invitation to Research",
      }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url ===
              `${INBOX}/44444444-4444-4444-4444-444444444444/accept`,
        ),
      ).toBe(true)
    })
    // No token anywhere on the wire: the session is the whole proof here.
    expect(requests.every((request) => !request.url.includes("token"))).toBe(
      true,
    )
  })

  it("puts decline behind a confirm that says the emailed link stops working", async () => {
    const requests = mockApi()
    renderPage()

    const user = userEvent.setup()
    await user.click(
      await screen.findByRole("button", {
        name: "Decline invitation to Research",
      }),
    )

    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/emailed\s+link stops working/),
    ).toBeVisible()
    // Opening the dialog alone must not have written anything.
    expect(requests.some((request) => request.method === "POST")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Decline invitation" }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.url ===
            `${INBOX}/44444444-4444-4444-4444-444444444444/decline`,
        ),
      ).toBe(true)
    })
  })

  it("drops the row once the write lands, because the list is invalidated", async () => {
    mockApi()
    renderPage()

    const user = userEvent.setup()
    await user.click(
      await screen.findByRole("button", {
        name: "Accept invitation to Research",
      }),
    )

    await waitFor(() => {
      expect(screen.queryByText("Research")).toBeNull()
    })
    expect(await screen.findByText("No invitations waiting")).toBeVisible()
  })

  it("reports a refusal instead of dropping the row", async () => {
    mockApi({ writeError: "invitation not found or already used" })
    renderPage()

    const user = userEvent.setup()
    await user.click(
      await screen.findByRole("button", {
        name: "Accept invitation to Research",
      }),
    )

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /not found or already used/,
    )
    // Still listed: the server refused, so nothing about this invitation moved.
    expect(screen.getByText("Research")).toBeVisible()
  })

  it("says nothing is waiting rather than rendering an empty list", async () => {
    mockApi({ waiting: [] })
    renderPage()

    expect(await screen.findByText("No invitations waiting")).toBeVisible()
    expect(
      screen.queryByRole("button", { name: /^Accept invitation to/ }),
    ).toBeNull()
  })

  it("keeps the two invitations separately actionable", async () => {
    const requests = mockApi({
      waiting: [
        pendingOrganizationInvitation(),
        pendingOrganizationInvitation({
          organization_member_id: "66666666-6666-6666-6666-666666666666",
          organization_name: "Platform",
          role: "viewer",
        }),
      ],
    })
    renderPage()

    const rows = await screen.findAllByRole("listitem")
    expect(rows).toHaveLength(2)

    const user = userEvent.setup()
    await user.click(
      within(rows[1]).getByRole("button", {
        name: "Accept invitation to Platform",
      }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.url ===
            `${INBOX}/66666666-6666-6666-6666-666666666666/accept`,
        ),
      ).toBe(true)
    })
  })

  it("reports a failed read rather than claiming nothing is waiting", async () => {
    // 404 rather than 500 because that is the case with a cause: a gateway
    // older than this bundle does not serve the route, and a hybrid one
    // answers 404 for every `/v1/organizations` path by design. It is also the
    // one the hook refuses to retry, so the banner is the first answer rather
    // than the fourth.
    mockApi({ listError: 404 })
    renderPage()

    expect(await screen.findByRole("alert")).toBeVisible()
    expect(screen.queryByText("No invitations waiting")).toBeNull()
  })
})
