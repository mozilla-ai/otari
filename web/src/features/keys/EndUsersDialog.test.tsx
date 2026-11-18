import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { EndUser, EndUserCount } from "@/client"
import { EndUsersDialog } from "@/features/keys/EndUsersDialog"
import { API_ROOT } from "@/shared/api/client"
import { apiKey, organizationContext } from "@/tests/fixtures"

const END_USERS_PATH = `${API_ROOT}/organizations/me/keys/key-1/end-users`

function endUser(overrides: Partial<EndUser> = {}): EndUser {
  return {
    user_id: "svc:alice",
    external_id: "alice",
    owner_user_id: "svc",
    budget_id: "org-free",
    blocked: false,
    spend: 0.25,
    reserved: 0,
    current_tokens: 120,
    current_requests: 3,
    budget_started_at: null,
    next_budget_reset_at: null,
    created_at: "2026-11-01T10:00:00+00:00",
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// An owner's view of an organization's service key: the member surface.
function mockApi(
  rows: EndUser[],
  count: EndUserCount = { count: rows.length, max_end_users: 10000 },
) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), "http://gateway.test")
    if (url.pathname === `${API_ROOT}/organizations/me`) {
      return jsonResponse(organizationContext({ deployment_operator: false }))
    }
    if (url.pathname === `${API_ROOT}/organizations/me/budgets`) {
      return jsonResponse({
        data: [
          {
            budget_id: "org-free",
            name: "Free tier",
            organization_id: "org-1",
          },
        ],
        count: 1,
      })
    }
    if (
      url.pathname === `${API_ROOT}/organizations/me/keys/key-1/end-user-count`
    ) {
      return jsonResponse(count)
    }
    if (url.pathname === END_USERS_PATH) {
      const externalId = url.searchParams.get("external_id")
      const skip = Number(url.searchParams.get("skip"))
      const limit = Number(url.searchParams.get("limit"))
      const matching = externalId
        ? rows.filter((row) => row.external_id === externalId)
        : rows
      return jsonResponse(matching.slice(skip, skip + limit))
    }
    return jsonResponse([])
  })
}

function renderDialog() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <EndUsersDialog
        apiKey={apiKey({
          id: "key-1",
          key_name: "mlpa",
          is_service_key: true,
          is_organization_service_key: true,
        })}
        canReadBudgets
        onClose={() => {}}
      />
    </QueryClientProvider>,
  )
}

describe("EndUsersDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists each end user with its budget's name and what it used", async () => {
    mockApi([
      endUser(),
      endUser({ user_id: "svc:bob", external_id: "bob", blocked: true }),
    ])
    renderDialog()

    const list = await screen.findByRole("list", { name: "End users of mlpa" })
    const items = within(list).getAllByRole("listitem")
    expect(items).toHaveLength(2)
    await waitFor(() =>
      expect(items[0]).toHaveTextContent(/Free tier · \$0\.25 · 3 requests/),
    )
    expect(within(items[1]).getByText("Blocked")).toBeInTheDocument()
  })

  it("finds one end user by its id", async () => {
    const fetchSpy = mockApi([
      endUser(),
      endUser({ user_id: "svc:bob", external_id: "bob" }),
    ])
    const user = userEvent.setup()
    renderDialog()

    await screen.findByRole("list", { name: "End users of mlpa" })
    await user.type(
      screen.getByRole("searchbox", { name: "Find an end user by id" }),
      "bob",
    )

    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(1))
    expect(screen.getByRole("listitem")).toHaveTextContent("bob")
    expect(
      fetchSpy.mock.calls.some(([input]) =>
        String(input).includes("external_id=bob"),
      ),
    ).toBe(true)
  })

  it("pages past the first 25 end users", async () => {
    const rows = Array.from({ length: 26 }, (_, index) =>
      endUser({ user_id: `svc:user-${index}`, external_id: `user-${index}` }),
    )
    const fetchSpy = mockApi(rows)
    const user = userEvent.setup()
    renderDialog()

    await screen.findByText("user-24")
    expect(screen.queryByText("user-25")).not.toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Next page, end users of mlpa" }),
    )

    expect(await screen.findByText("user-25")).toBeInTheDocument()
    expect(screen.queryByText("user-0")).not.toBeInTheDocument()
    expect(
      fetchSpy.mock.calls.some(([input]) =>
        String(input).includes("skip=25&limit=26"),
      ),
    ).toBe(true)
  })

  it("shows the end users the cap counts against the cap, and when it is full", async () => {
    mockApi([endUser(), endUser({ user_id: "svc:bob", external_id: "bob" })], {
      count: 2,
      max_end_users: 2,
    })
    renderDialog()

    expect(await screen.findByText("2 of 2 end users")).toBeInTheDocument()
    expect(screen.getByText("Full")).toBeInTheDocument()
  })

  it("says a key without a cap has none", async () => {
    mockApi([endUser()], { count: 1, max_end_users: null })
    renderDialog()

    expect(await screen.findByText("1 end user, no cap")).toBeInTheDocument()
    expect(screen.queryByText("Full")).not.toBeInTheDocument()
  })

  it("says the cap could not be read, and reads it again on Retry", async () => {
    const fetchSpy = mockApi([endUser()])
    const answer = fetchSpy.getMockImplementation()!
    let failing = true
    fetchSpy.mockImplementation(async (input, init) => {
      if (failing && String(input).includes("/end-user-count")) {
        return jsonResponse({ detail: "Database error" }, 500)
      }
      return answer(input, init)
    })
    const user = userEvent.setup()
    renderDialog()

    expect(
      await screen.findByText("Could not read the cap"),
    ).toBeInTheDocument()
    failing = false
    await user.click(screen.getByRole("button", { name: "Retry" }))

    expect(await screen.findByText("1 of 10,000 end users")).toBeInTheDocument()
  })

  it("says when the key has no end users", async () => {
    mockApi([])
    renderDialog()

    expect(await screen.findByText("No end users yet.")).toBeInTheDocument()
  })
})
