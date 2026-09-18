import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap } from "@/client"
import { KeysPage } from "@/features/keys/KeysPage"
import { API_ROOT } from "@/shared/api/client"
import { apiKey, bootstrap } from "@/tests/fixtures"
import {
  chooseAction,
  mockApi,
  NEW_SECRET,
  renderPage,
  submitTheCreateDialog,
} from "@/tests/keys"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("KeysPage secret reveal", () => {
  it("conceals a new key and its snippets until explicitly revealed", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByLabelText("Name"), "deploy-key")
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    const reveal = await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    expect(within(reveal).getByLabelText("Secret key")).toHaveValue(
      "gw-NEWSECR••••••••0000",
    )
    expect(
      (within(reveal).getByLabelText("curl") as HTMLTextAreaElement).value,
    ).not.toContain(NEW_SECRET)
    expect(
      (
        within(reveal).getByLabelText(
          "Python (Otari SDK)",
        ) as HTMLTextAreaElement
      ).value,
    ).not.toContain(NEW_SECRET)
    await user.click(
      within(reveal).getByRole("button", { name: "Show Secret key" }),
    )
    expect(within(reveal).getByLabelText("Secret key")).toHaveValue(NEW_SECRET)
    const curl = within(reveal).getByLabelText("curl") as HTMLTextAreaElement
    const python = within(reveal).getByLabelText(
      "Python (Otari SDK)",
    ) as HTMLTextAreaElement
    expect(curl.value).toContain(`Otari-Key: ${NEW_SECRET}`)
    expect(curl.value).toContain(
      `${window.location.origin}${API_ROOT}/chat/completions`,
    )
    expect(python.value).toContain(NEW_SECRET)

    // One credential, one reveal: concealing the key conceals the requests that
    // carry it, rather than leaving it in plain sight twice over.
    await user.click(
      within(reveal).getByRole("button", { name: "Hide Secret key" }),
    )
    expect(within(reveal).getByLabelText("Secret key")).not.toHaveValue(
      NEW_SECRET,
    )
    expect(curl.value).not.toContain(NEW_SECRET)
    expect(python.value).not.toContain(NEW_SECRET)

    // And the affordance is on each field, not only on the key: the snippet's
    // own toggle brings all three back (otari-ai#2111).
    await user.click(within(reveal).getByRole("button", { name: "Show curl" }))
    expect(within(reveal).getByLabelText("Secret key")).toHaveValue(NEW_SECRET)
    expect(curl.value).toContain(NEW_SECRET)
    expect(python.value).toContain(NEW_SECRET)

    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )

    // After closing, only the fingerprint remains.
    expect(
      screen.queryByRole("alert", { name: /API key created|New secret for/ }),
    ).not.toBeInTheDocument()
    expect(
      await screen.findByText(
        `${NEW_SECRET.slice(0, 10)}…${NEW_SECRET.slice(-4)}`,
      ),
    ).toBeInTheDocument()
    expect(document.body.textContent).not.toContain(NEW_SECRET)
  })

  /** Mint a key and open the one-time reveal, which is where the snippets live. */
  async function revealANewKey(deployment?: DeploymentBootstrap) {
    mockApi({ keys: [] })
    const person = userEvent.setup()
    renderPage(<KeysPage />, deployment)

    await screen.findByText("No API keys yet")
    await person.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await person.type(screen.getByLabelText("Name"), "deploy-key")
    await person.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await person.keyboard("{Escape}")
    await person.click(screen.getByRole("button", { name: "Create key" }))

    return within(
      await screen.findByRole("alert", {
        name: /API key created|New secret for/,
      }),
    )
  }

  it("sends the snippet at the data plane a hosted deployment published", async () => {
    // The control plane serves this dashboard and is deliberately not where
    // inference belongs (otari#822), so the origin is the one address the
    // snippet must not name.
    const dialog = await revealANewKey(
      bootstrap({
        deployment_type: "hosted",
        data_plane_url: "https://gateway.otari.ai",
      }),
    )

    const curl = dialog.getByLabelText("curl") as HTMLTextAreaElement
    expect(curl.value).toContain(
      `https://gateway.otari.ai${API_ROOT}/chat/completions`,
    )
    expect(curl.value).not.toContain(window.location.origin)
  })

  it("shows no snippet when a hosted deployment published no data plane", async () => {
    // Withheld rather than aimed at this host: a placeholder would be a URL
    // nobody reading it could replace, and the origin would be the bug itself.
    const dialog = await revealANewKey(
      bootstrap({ deployment_type: "hosted", data_plane_url: null }),
    )

    expect(dialog.getByLabelText("Secret key")).toBeInTheDocument()
    expect(dialog.queryByLabelText("curl")).not.toBeInTheDocument()
    expect(
      dialog.getByText(/has not published the gateway address/),
    ).toBeInTheDocument()
  })

  it("keeps the reveal up through a stray Escape; only the save button dismisses it", async () => {
    // The reveal is a strip on the page now rather than a modal, so there is no
    // Esc handler to suppress and no backdrop to click. What still has to hold
    // is the thing the modal was protecting: a one-time secret cannot be lost
    // to a keystroke aimed at something else.
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    await user.keyboard("{Escape}")
    expect(
      screen.getByRole("alert", { name: /API key created|New secret for/ }),
    ).toBeInTheDocument()

    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )
    expect(
      screen.queryByRole("alert", { name: /API key created|New secret for/ }),
    ).not.toBeInTheDocument()
  })

  // The reveal is a modal again, and the objection its docstring used to raise
  // is answered rather than ignored: Escape and the backdrop work everywhere in
  // this dialog EXCEPT here, where the content cannot be recovered. So the press
  // that must not dismiss it is the backdrop, which is the only "elsewhere" a
  // modal has; the page behind is deliberately out of the accessibility tree now
  // and there is nothing on it to click.

  it("keeps the reveal up through a press on the backdrop", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    const backdrop = document.querySelector('[class*="modal__backdrop"]')
    expect(backdrop).not.toBeNull()
    await user.click(backdrop as Element)
    expect(
      screen.getByRole("alert", { name: /API key created|New secret for/ }),
    ).toBeInTheDocument()

    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )
    await waitFor(() =>
      expect(
        screen.queryByRole("alert", {
          name: /API key created|New secret for/,
        }),
      ).toBeNull(),
    )
  })

  it("returns focus to the page's create action when the reveal closes", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )

    // The form that opened the reveal is gone by now, so nothing else has a
    // claim on focus and it would otherwise be left on <body>.
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Create key" }),
      ),
    )
  })

  it("confirms Copied when the clipboard API is available", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    // Install after userEvent.setup(), which otherwise replaces navigator.clipboard
    // with its own stub.
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    })
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    const reveal = await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    await user.click(
      within(reveal).getByRole("button", { name: "Copy Secret key" }),
    )

    // The key reached the clipboard and never the screen.
    expect(writeText).toHaveBeenCalledWith(NEW_SECRET)
    expect(
      within(reveal).queryByDisplayValue(NEW_SECRET),
    ).not.toBeInTheDocument()
    expect(
      await within(reveal).findByText("Copied to clipboard."),
    ).toBeInTheDocument()
  })

  it("hands over the secret with no way for the form to skip it", async () => {
    // The one chance to read the key is not something the form can waive. The
    // only "Create another" is on the secret step, where it is reached by
    // having been shown the key first.
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).queryByLabelText("Create another"),
    ).not.toBeInTheDocument()
    await user.type(screen.getByLabelText("Name"), "first-key")
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    expect(
      await screen.findByRole("alert", { name: /API key created/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    ).toBeInTheDocument()
  })

  it("shows a regenerated secret in the same dialog a created one arrives in", async () => {
    // Create and regenerate hand over the same thing, so they hand it over the
    // same way. Before this they were a dialog and a full-width strip on one
    // page.
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await chooseAction(user, row, "Regenerate")
    const confirm = await screen.findByRole("alertdialog")
    await user.click(
      within(confirm).getByRole("button", { name: "Regenerate" }),
    )

    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("alert", { name: /New secret for ci-bot/ }),
    ).toBeInTheDocument()
    // No form step and nothing to create again: the key already exists.
    expect(
      within(dialog).queryByRole("button", { name: "Create another" }),
    ).not.toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: /I.?ve saved this key/ }),
    ).toBeInTheDocument()
  })
})
