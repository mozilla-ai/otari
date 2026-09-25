import {
  onlineManager,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { FeedbackDialog } from "./FeedbackDialog"

const SEND = /^Send to the Otari team/
const NOT_SENT =
  "That didn’t reach us. Your message is still here; send it again."

function setup() {
  const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
  vi.stubGlobal("fetch", fetch)
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  function Harness() {
    const [open, setOpen] = useState(true)
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>
          Open feedback
        </button>
        <FeedbackDialog isOpen={open} onOpenChange={setOpen} />
      </>
    )
  }
  render(
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>,
  )
  return { fetch, user: userEvent.setup() }
}

const field = () => screen.getByRole("textbox", { name: "Feedback" })

afterEach(() => {
  onlineManager.setOnline(true)
  vi.unstubAllGlobals()
})

describe("FeedbackDialog", () => {
  it("sends the trimmed message alone and thanks in place, with no button", async () => {
    const { fetch, user } = setup()
    await user.type(field(), "  Research budgets  ")
    expect(fetch).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: SEND }))
    expect(
      await screen.findByRole("heading", { name: "Thank you!" }),
    ).toBeInTheDocument()
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(fetch.mock.calls[0][0]).toBe("/api/v1/feedback")
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      message: "Research budgets",
    })
    const dialog = screen.getByRole("dialog", { name: "Thank you!" })
    await waitFor(() => expect(dialog).toHaveFocus())
    expect(dialog).toHaveAccessibleDescription(
      "Every message reaches the people building Otari. We read them all, and they shape what comes next.",
    )
    // The phone sheet's close control is the frame's only button.
    expect(
      [...dialog.querySelectorAll("button")].map((b) =>
        b.getAttribute("aria-label"),
      ),
    ).toEqual(["Close"])
    await user.keyboard("{Escape}")
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    await user.click(screen.getByRole("button", { name: "Open feedback" }))
    expect(
      await screen.findByRole("textbox", { name: "Feedback" }),
    ).toHaveValue("")
  })

  it("sends on Cmd or Ctrl+Enter, and Enter alone is a newline", async () => {
    const { fetch, user } = setup()
    await user.type(field(), "One{Enter}Two")
    expect(field()).toHaveValue("One\nTwo")
    expect(fetch).not.toHaveBeenCalled()
    await user.keyboard("{Control>}{Enter}{/Control}")
    await screen.findByRole("heading", { name: "Thank you!" })
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      message: "One\nTwo",
    })
  })

  it("answers an empty send beside the button, which stays live", async () => {
    const { fetch, user } = setup()
    // Mounted before there is anything to say, so the reason is announced.
    expect(screen.getByRole("status")).toBeEmptyDOMElement()
    await user.type(field(), "   ")
    await user.click(screen.getByRole("button", { name: SEND }))
    expect(screen.getByRole("status")).toHaveTextContent(
      "Write something first.",
    )
    expect(field()).toHaveAttribute("aria-invalid", "true")
    expect(screen.getByRole("button", { name: SEND })).toBeEnabled()
    expect(fetch).not.toHaveBeenCalled()
    await user.type(field(), "x")
    expect(screen.getByRole("status")).toBeEmptyDOMElement()
    expect(field()).not.toHaveAttribute("aria-invalid")
  })

  it.each([503, 429])(
    "keeps the draft on a %i and says so in one sentence, then retries by hand",
    async (status) => {
      const { fetch, user } = setup()
      fetch.mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "Upstream detail" }), {
          status,
        }),
      )
      await user.type(field(), "Idea")
      await user.click(screen.getByRole("button", { name: SEND }))
      expect(await screen.findByRole("alert")).toHaveTextContent(NOT_SENT)
      expect(field()).toHaveValue("Idea")
      expect(fetch).toHaveBeenCalledTimes(1)
      await user.click(screen.getByRole("button", { name: SEND }))
      await screen.findByRole("heading", { name: "Thank you!" })
      expect(fetch).toHaveBeenCalledTimes(2)
    },
  )

  it("does not queue an offline send for reconnection", async () => {
    onlineManager.setOnline(false)
    const { fetch, user } = setup()
    fetch.mockRejectedValue(new TypeError("Offline"))
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    await screen.findByRole("alert")
    onlineManager.setOnline(true)
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(field()).toHaveValue("Idea")
  })

  it("guards Escape on a draft and clears it only on Discard", async () => {
    const { fetch, user } = setup()
    expect(
      screen.queryByRole("button", { name: "Cancel" }),
    ).not.toBeInTheDocument()
    await user.type(field(), "Unsent idea")
    await user.keyboard("{Escape}")
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Keep editing" }))
    expect(field()).toHaveValue("Unsent idea")
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    await user.click(screen.getByRole("button", { name: "Open feedback" }))
    expect(
      await screen.findByRole("textbox", { name: "Feedback" }),
    ).toHaveValue("")
    expect(fetch).not.toHaveBeenCalled()
  })

  it("ignores Escape and holds the field read-only while a send is in flight", async () => {
    const { fetch, user } = setup()
    let answer: (response: Response) => void = () => {}
    fetch.mockReturnValueOnce(
      new Promise<Response>((resolve) => {
        answer = resolve
      }),
    )
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    await waitFor(() => expect(field()).toHaveAttribute("readonly"))
    await user.keyboard("{Escape}")
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument()
    answer(new Response(null, { status: 204 }))
    await screen.findByRole("heading", { name: "Thank you!" })
  })

  it("caps the field at the gateway's 4,000", () => {
    setup()
    expect(field()).toHaveAttribute("maxlength", "4000")
  })
})
