import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { KeysPage } from "@/features/keys/KeysPage"
import { API_ROOT } from "@/shared/api/client"
import { apiKey } from "@/tests/fixtures"
import { chooseAction, KEYS_URL, mockApi, renderPage } from "@/tests/keys"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("KeysPage as a member", () => {
  it("lists through the member surface, without the Owner column or operator links", async () => {
    const fetchMock = mockApi({
      deploymentOperator: false,
      keys: [apiKey({ id: "key-1", key_name: "mine" })],
    })
    renderPage(<KeysPage />)

    const row = (await screen.findByText("mine")).closest("tr")!
    expect(within(row).getByText("Active")).toBeInTheDocument()

    // The read went to the member surface, and never to the operator one.
    const listCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => KEYS_URL.test(u))
    expect(listCalls.length).toBeGreaterThan(0)
    for (const u of listCalls) {
      expect(u).toContain(`${API_ROOT}/organizations/me/keys`)
    }

    // Every key here is the caller's own, so no Owner column; and the pages
    // the operator paragraph links to would refuse a member.
    expect(
      screen.queryByRole("columnheader", { name: "Owner" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("link", { name: /Spend & budgets/ }),
    ).not.toBeInTheDocument()
  })

  it("creates a key with no owner picker and no budget exemption", async () => {
    const fetchMock = mockApi({ deploymentOperator: false, keys: [] })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await usr.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )

    // No owner to pick: the key is the caller's own, and Create does not wait
    // for one.
    expect(screen.queryByPlaceholderText(/Pick a user/)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create key" })).toBeEnabled()

    await usr.click(screen.getByRole("button", { name: "Advanced" }))
    expect(
      screen.queryByLabelText("Exempt from budget"),
    ).not.toBeInTheDocument()

    await usr.type(screen.getByPlaceholderText("ci-bot"), "my-key")
    await usr.click(screen.getByRole("button", { name: "Create key" }))
    const reveal = await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    expect(within(reveal).getByLabelText("Secret key")).toHaveValue(
      "gw-NEWSECR••••••••0000",
    )

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/organizations/me/keys`) &&
        (init?.method ?? "") === "POST",
    )
    expect(post).toBeDefined()
    const body = JSON.parse(String(post?.[1]?.body))
    expect(body.key_name).toBe("my-key")
    // The member body carries neither escalation field.
    expect(body).not.toHaveProperty("user_id")
    expect(body).not.toHaveProperty("exclude_from_budget")
  })

  it("edits through the member surface, with no budget exemption to send", async () => {
    const fetchMock = mockApi({
      deploymentOperator: false,
      keys: [apiKey({ id: "key-1", key_name: "mine" })],
    })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("mine")).closest("tr")!
    await chooseAction(usr, row, "Edit")
    expect(
      screen.queryByLabelText("Exempt from budget"),
    ).not.toBeInTheDocument()

    await usr.click(screen.getByRole("button", { name: "Save" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/organizations/me/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    expect(patch).toBeDefined()
    expect(JSON.parse(String(patch?.[1]?.body))).not.toHaveProperty(
      "exclude_from_budget",
    )
  })
})
