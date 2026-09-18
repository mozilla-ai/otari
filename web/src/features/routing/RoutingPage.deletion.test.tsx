import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import {
  CHAIN,
  createTrigger,
  LEARNED,
  mockApi,
  policy,
  renderPage,
} from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage deletion", () => {
  it("names the policy in a confirm dialog before deleting it", async () => {
    // otari-ai#2110: the confirmation used to arm inside the row, where it read
    // as part of the table rather than as a decision. It is a modal now, and
    // the policy it is about has to be named in it: the row is behind the
    // backdrop, so the name on the row is no longer the operator's reference.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/^fast stops resolving/)).toBeVisible()
    // Nothing is sent by opening it.
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Delete policy" }),
    )

    const deletes = calls.filter((call) => call.method === "DELETE")
    expect(deletes).toHaveLength(1)
    expect(deletes[0].url).toContain(`${API_ROOT}/routing/policies/fast`)
  })

  it("deletes nothing when the confirm dialog is cancelled", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))

    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    expect(screen.getByText("fast")).toBeInTheDocument()
  })

  it("returns focus to the page's action when the empty state's dialog closes", async () => {
    // Creating the first policy fills the table, so the empty state unmounts and
    // the node react-aria stored for focus restoration is gone: focus resets to
    // `document.body` and the next Tab starts at the top of the document. The
    // page's own trigger is where it lands instead.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const empty = (
      await screen.findByRole("heading", { name: "No routing policies yet" })
    ).closest("div")!.parentElement!
    await user.click(
      within(empty).getByRole("button", { name: "Create your first policy" }),
    )
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    await user.keyboard("{Escape}")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const trigger = await createTrigger()
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it("reports a refused delete inside the dialog, leaving the row", async () => {
    // The page banner no longer carries this: the operator is looking at the
    // modal, and a message behind the backdrop is a message they do not read.
    mockApi([policy("fast", CHAIN)], "http://guardrails:8000", [], {
      deleteBody: { status: 409, detail: "fast is referenced by an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Delete policy" }),
    )

    expect(
      await within(dialog).findByText(/referenced by an alias/),
    ).toBeVisible()
    // Still open, so the operator can retry or back out rather than being
    // returned to a table that looks unchanged for no stated reason.
    expect(screen.getByRole("alertdialog")).toBeInTheDocument()
    expect(screen.getByText("fast")).toBeInTheDocument()
  })

  it("does not greet the next row's confirm with the last row's refusal", async () => {
    // The mutation holds its error until the next call, and the dialog reads it,
    // so without clearing it on close the second row opens already reporting a
    // failure that was about the first.
    mockApi([policy("fast", CHAIN), policy("smart", LEARNED)], undefined, [], {
      deleteBody: { status: 409, detail: "fast is referenced by an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const fast = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(fast).getByRole("button", { name: "Delete" }))
    const first = await screen.findByRole("alertdialog")
    await user.click(
      within(first).getByRole("button", { name: "Delete policy" }),
    )
    await within(first).findByText(/referenced by an alias/)
    await user.click(within(first).getByRole("button", { name: "Cancel" }))

    const smart = screen.getByText("smart").closest("tr")!
    await user.click(within(smart).getByRole("button", { name: "Delete" }))

    const second = await screen.findByRole("alertdialog")
    expect(within(second).getByText(/^smart stops resolving/)).toBeVisible()
    expect(within(second).queryByText(/referenced by an alias/)).toBeNull()
  })

  it("deletes an alias through the alias endpoint, not the policy one", async () => {
    const { calls } = mockApi([], "http://guardrails:8000", [
      {
        name: "legacy",
        target: "openai:gpt-4o-mini",
        source: "stored",
        user_id: null,
      },
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete alias",
      }),
    )

    const deletes = calls.filter((call) => call.method === "DELETE")
    expect(deletes).toHaveLength(1)
    // An alias still lives in model_aliases; deleting it as a policy would 404 and
    // leave the row in place.
    expect(deletes[0].url).toContain(`${API_ROOT}/aliases/legacy`)
  })

  it("will not let an alias grow options an alias cannot hold", async () => {
    mockApi([], "http://guardrails:8000", [
      {
        name: "legacy",
        target: "openai:gpt-4o-mini",
        source: "stored",
        user_id: null,
      },
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(
      await screen.findByRole("button", { name: /Add a fallback chain/ }),
    )

    // Saving it as a policy would leave the alias row behind under the same name,
    // and the API refuses that collision, so the form says so instead of failing.
    expect(screen.getByText(/An alias holds one target/)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
  })
})
