import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AccessTokensCard } from "@/features/account/AccessTokensCard"
import { API_ROOT } from "@/shared/api/client"
import { AppProviders } from "@/tests/providers"
import { pressRowAction } from "@/tests/rowMenu"

const TOKEN = {
  id: "22222222-2222-2222-2222-222222222222",
  name: "Claude Code",
  token_suffix: "AbCd",
  created_at: "2026-11-20T10:00:00Z",
  last_used_at: null,
  expires_at: null,
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(routes: Record<string, () => Response>) {
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : String(input)
      const method = init?.method ?? "GET"
      const handler = routes[`${method} ${url}`]
      if (!handler) {
        throw new Error(`unexpected request: ${method} ${url}`)
      }
      return Promise.resolve(handler())
    })
}

const LIST = `GET ${API_ROOT}/auth/access-tokens`
const CREATE = `POST ${API_ROOT}/auth/access-tokens`
const DELETE = `DELETE ${API_ROOT}/auth/access-tokens/${TOKEN.id}`

function renderCard() {
  return render(
    <AppProviders>
      <AccessTokensCard />
    </AppProviders>,
  )
}

describe("AccessTokensCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists the caller's tokens by name and suffix", async () => {
    mockApi({ [LIST]: () => jsonResponse({ data: [TOKEN] }) })
    renderCard()

    expect(await screen.findByText("Claude Code")).toBeInTheDocument()
    expect(screen.getByText("…AbCd")).toBeInTheDocument()
    expect(screen.getByText(/Never used/)).toBeInTheDocument()
    expect(screen.getByText(/Never expires/)).toBeInTheDocument()
  })

  it("says so when there are none", async () => {
    mockApi({ [LIST]: () => jsonResponse({ data: [] }) })
    renderCard()

    expect(
      await screen.findByText("You have no access tokens."),
    ).toBeInTheDocument()
  })

  it("creates a token and hands it over once", async () => {
    const fetchMock = mockApi({
      [LIST]: () => jsonResponse({ data: [] }),
      [CREATE]: () =>
        jsonResponse({ ...TOKEN, token: "otari-pat-secret-AbCd" }, 201),
    })
    const user = userEvent.setup()
    renderCard()

    await screen.findByText("You have no access tokens.")
    await user.click(screen.getByRole("button", { name: "Create token" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText("Name"), "Claude Code")
    await user.click(
      within(dialog).getByRole("button", { name: "Create token" }),
    )

    const created = await screen.findByRole("alert", {
      name: "Access token created",
    })
    expect(within(created).getByText(/shown only once/)).toBeInTheDocument()
    expect(within(created).getByLabelText("MCP server")).toHaveValue(
      `${window.location.origin}/mcp`,
    )
    const posted = fetchMock.mock.calls.find(
      ([url, init]) =>
        String(url) === `${API_ROOT}/auth/access-tokens` &&
        (init as RequestInit)?.method === "POST",
    )
    if (!posted) {
      throw new Error("the create call was never made")
    }
    expect(JSON.parse(String((posted[1] as RequestInit).body))).toEqual({
      name: "Claude Code",
      expiration: "90-days",
    })

    await user.click(
      screen.getByRole("button", { name: "I’ve saved this token" }),
    )
    await waitFor(() =>
      expect(
        screen.queryByRole("alert", { name: "Access token created" }),
      ).not.toBeInTheDocument(),
    )
  })

  it("deletes a token after confirming", async () => {
    let deleted = false
    const fetchMock = mockApi({
      [LIST]: () => jsonResponse({ data: deleted ? [] : [TOKEN] }),
      [DELETE]: () => {
        deleted = true
        return new Response(null, { status: 204 })
      },
    })
    const user = userEvent.setup()
    renderCard()

    await screen.findByText("Claude Code")
    await pressRowAction(user, "Delete…")
    await user.click(
      within(screen.getByRole("alertdialog")).getByRole("button", {
        name: "Delete",
      }),
    )

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url) === `${API_ROOT}/auth/access-tokens/${TOKEN.id}` &&
            (init as RequestInit)?.method === "DELETE",
        ),
      ).toBe(true),
    )
    await waitFor(() =>
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument(),
    )
    expect(
      await screen.findByText("You have no access tokens."),
    ).toBeInTheDocument()
  })
})
