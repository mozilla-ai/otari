import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { KeysPage } from "@/features/keys/KeysPage"
import { API_ROOT } from "@/shared/api/client"
import { apiKey } from "@/tests/fixtures"
import { chooseAction, mockApi, REGEN_SECRET, renderPage } from "@/tests/keys"
import { pickOption } from "@/tests/select"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("KeysPage editing and key lifecycle", () => {
  it("opens actions with the keyboard and returns focus after cancelling regeneration", async () => {
    const fetchMock = mockApi({ keys: [apiKey()] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)
    const row = (await screen.findByText("ci-bot")).closest("tr")!
    const trigger = within(row).getByRole("button", {
      name: "Actions for ci-bot",
    })
    // Open it once with the pointer first. Until an overlay has been opened in
    // this document, ArrowDown on the trigger reaches the table's own row
    // navigation instead of the menu, and focus lands on the next row. The
    // whole flow used to sit after the secret-reveal specs, which opened one
    // for their own reasons and left this pressing keys on a warmed document.
    await user.click(trigger)
    await user.keyboard("{Escape}")
    await waitFor(() => expect(screen.queryByRole("menu")).toBeNull())

    trigger.focus()
    await user.keyboard("{ArrowDown}")
    const menu = await screen.findByRole("menu")
    expect(
      within(menu).getByRole("menuitem", { name: /^Delete/ }),
    ).toHaveAttribute("aria-disabled", "true")
    await user.click(
      within(menu).getByRole("menuitem", { name: /^Regenerate/ }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/stops working immediately/),
    ).toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(trigger).toHaveFocus())
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).endsWith("/rotate")),
    ).toBe(false)
  })

  it("disables an active key via PATCH, then offers permanent delete", async () => {
    const fetchMock = mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    // An active key refuses Delete, in the menu and with its reason
    // (require-disable-first).
    await user.click(
      within(row).getByRole("button", { name: "Actions for ci-bot" }),
    )
    const refused = await screen.findByRole("menuitem", { name: /^Delete/ })
    expect(refused).toHaveAttribute("aria-disabled", "true")
    expect(refused).toHaveTextContent("Disable it first")
    await user.keyboard("{Escape}")

    await chooseAction(user, row, "Disable")

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ is_active: false })

    const disabledRow = (await screen.findByText("Disabled")).closest("tr")!
    await user.click(
      within(disabledRow).getByRole("button", { name: "Actions for ci-bot" }),
    )
    expect(
      await screen.findByRole("menuitem", { name: /^Delete/ }),
    ).not.toHaveAttribute("aria-disabled", "true")
  })

  it("regenerates a secret after an explicit confirm", async () => {
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Regenerate")
    const confirm = await screen.findByRole("alertdialog")
    expect(within(confirm).getByText("ci-bot")).toBeInTheDocument()
    await user.click(
      within(confirm).getByRole("button", { name: "Regenerate" }),
    )

    const reveal = await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    expect(within(reveal).getByLabelText("Secret key")).toHaveValue(
      "gw-REGEN00••••••••0000",
    )
    expect(
      (within(reveal).getByLabelText("curl") as HTMLTextAreaElement).value,
    ).not.toContain(REGEN_SECRET)
  })

  it("permanently deletes a disabled key after confirm", async () => {
    const fetchMock = mockApi({
      keys: [apiKey({ id: "key-1", key_name: "legacy", is_active: false })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    await chooseAction(user, row, "Delete")
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/unlinks its usage history/),
    ).toBeInTheDocument()
    // The key is named in the dialog, so the operator is not confirming against
    // a row they can no longer see.
    expect(within(dialog).getByText("legacy")).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Delete permanently" }),
    )

    const del = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "DELETE",
    )
    expect(del).toBeDefined()
    expect(screen.queryByText("legacy")).not.toBeInTheDocument()
  })

  it("opens the edit form in a dialog, naming the key it is about", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Edit")

    // A dialog rather than a band above the table: the row keeps its place, and
    // the page under it does not shift by the height of a form (otari-ai#2125).
    const dialog = await screen.findByRole("dialog", { name: "Edit key" })
    expect(within(dialog).getByText("ci-bot")).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Save" }),
    ).toBeInTheDocument()
    expect(within(dialog).getByLabelText("Name")).toHaveValue("ci-bot")
    await user.keyboard("{Escape}")
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    expect(
      within(row).getByRole("button", { name: "Actions for ci-bot" }),
    ).toHaveFocus()
  })

  it("asks before discarding an edited field", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Edit")
    await user.type(await screen.findByLabelText("Name"), "-2")
    await user.keyboard("{Escape}")

    // The guard, not the exit: an edit form seeds from the row, so dirty has to
    // mean "differs from the key" rather than "is not empty".
    expect(screen.getByRole("button", { name: "Keep editing" })).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
  })

  it("toggles exclude_from_budget on an existing key via PATCH", async () => {
    const fetchMock = mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "ci-bot", exclude_from_budget: false }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Edit")
    await user.click(await screen.findByLabelText("Exempt from budget"))
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body)).exclude_from_budget).toBe(true)
  })

  it("sets reject_user_mismatch on an existing key via PATCH", async () => {
    const fetchMock = mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "ci-bot", reject_user_mismatch: null }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Edit")
    await pickOption(user, "Mismatched user field", "Always accept")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body)).reject_user_mismatch).toBe(
      false,
    )
  })

  it("clears a key's override back to inheriting via PATCH", async () => {
    const fetchMock = mockApi({
      keys: [
        apiKey({
          id: "key-1",
          key_name: "ci-bot",
          reject_user_mismatch: false,
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Edit")
    await pickOption(
      user,
      "Mismatched user field",
      "Use the deployment setting (default)",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    // An explicit null is what clears the override; omitting it would leave it set.
    expect(JSON.parse(String(patch?.[1]?.body)).reject_user_mismatch).toBeNull()
  })

  it("resets the edit form when switching to a different key row", async () => {
    mockApi({
      keys: [
        apiKey({ id: "k1", key_name: "alpha" }),
        apiKey({ id: "k2", key_name: "bravo" }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const alphaRow = (await screen.findByText("alpha")).closest("tr")!
    await chooseAction(user, alphaRow, "Edit")
    expect(await screen.findByLabelText("Name")).toHaveValue("alpha")
    await user.click(screen.getByRole("button", { name: "Cancel" }))

    // The next key must open on its own values; a form that survived the first
    // row would keep "alpha" and PATCH the wrong key.
    const bravoRow = screen.getByText("bravo").closest("tr")!
    await chooseAction(user, bravoRow, "Edit")
    expect(await screen.findByLabelText("Name")).toHaveValue("bravo")
  })

  it("clicking a row action does not also open the edit form", async () => {
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Disable")

    expect(
      screen.queryByRole("button", { name: "Save" }),
    ).not.toBeInTheDocument()
  })
})
