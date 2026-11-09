import {
  onlineManager,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { FeedbackDialog } from "./FeedbackDialog"

const SEND = /^Send to the Otari team/
const NOT_SENT =
  "That didn’t reach us. Your message is still here; send it again."

function setup({ email }: { email?: string } = {}) {
  const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
  vi.stubGlobal("fetch", fetch)
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  let showEmail: (next: string) => void = () => {}
  function Harness() {
    const [open, setOpen] = useState(true)
    const [shown, setShown] = useState(email)
    showEmail = setShown
    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>
          Open feedback
        </button>
        <FeedbackDialog isOpen={open} onOpenChange={setOpen} email={shown} />
      </>
    )
  }
  render(
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>,
  )
  return {
    fetch,
    user: userEvent.setup(),
    showEmail: (next: string) => act(() => showEmail(next)),
  }
}

const field = () => screen.getByRole("textbox", { name: "Contact" })

afterEach(() => {
  onlineManager.setOnline(true)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
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
    expect(dialog).toHaveAccessibleDescription("Our team has your message.")
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
    expect(await screen.findByRole("textbox", { name: "Contact" })).toHaveValue(
      "",
    )
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
    expect(await screen.findByRole("textbox", { name: "Contact" })).toHaveValue(
      "",
    )
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
  const consent = (email: string) =>
    `By sending, you agree we can reply to ${email} about this message only.`

  it("names no address when the account has none, and sends the message alone", async () => {
    const { fetch, user } = setup()
    expect(screen.queryByText(/By sending/)).not.toBeInTheDocument()
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    await screen.findByRole("heading", { name: "Thank you!" })
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ message: "Idea" })
  })

  it.each([
    ["no dot in the domain", "ops@intranet"],
    ["past the receiver's 254", `${"a".repeat(243)}@example.com`],
    ["a bidirectional override", "ada@example.com‮moc"],
  ])(
    "names no address the gateway would refuse to attach: %s",
    async (_, email) => {
      const { fetch, user } = setup({ email })
      expect(screen.queryByText(/By sending/)).not.toBeInTheDocument()
      await user.type(field(), "Idea")
      await user.click(screen.getByRole("button", { name: SEND }))
      await screen.findByRole("heading", { name: "Thank you!" })
      expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
        message: "Idea",
      })
    },
  )

  it("says where replies go, sends that address, and the thanks name it", async () => {
    const { fetch, user } = setup({ email: "ada@example.com" })
    expect(screen.getByText(consent("ada@example.com"))).toBeInTheDocument()
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    await screen.findByRole("heading", { name: "Thank you!" })
    // The address the line named, never trusted as such: the gateway sends the
    // session's own, and only when it is this one.
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      message: "Idea",
      share_email: "ada@example.com",
    })
    expect(
      screen.getByRole("dialog", { name: "Thank you!" }),
    ).toHaveAccessibleDescription("Our team will reply to ada@example.com.")
  })

  it("names and sends the address showing when Send is pressed", async () => {
    const { fetch, user, showEmail } = setup({ email: "ada@example.com" })
    await user.type(field(), "Idea")
    // Another tab signed in as someone else, and the caller query caught up.
    showEmail("grace@example.com")
    expect(screen.getByText(consent("grace@example.com"))).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: SEND }))
    await screen.findByRole("heading", { name: "Thank you!" })
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      message: "Idea",
      share_email: "grace@example.com",
    })
  })

  it("thanks with the address the send carried, even if another shows meanwhile", async () => {
    const { fetch, user, showEmail } = setup({ email: "ada@example.com" })
    let answer: (response: Response) => void = () => {}
    fetch.mockReturnValueOnce(
      new Promise<Response>((resolve) => {
        answer = resolve
      }),
    )
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    showEmail("grace@example.com")
    answer(new Response(null, { status: 204 }))
    await screen.findByRole("heading", { name: "Thank you!" })
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      message: "Idea",
      share_email: "ada@example.com",
    })
    expect(
      screen.getByRole("dialog", { name: "Thank you!" }),
    ).toHaveAccessibleDescription("Our team will reply to ada@example.com.")
  })

  it("sends on Cmd+Enter from the field, as the button promises", async () => {
    const { fetch, user } = setup({ email: "ada@example.com" })
    await user.type(field(), "Idea")
    // The raw keydown: user-event's Enter would add a line to the textarea and
    // hide a shortcut that never fired.
    fireEvent.keyDown(field(), { key: "Enter", metaKey: true })
    await screen.findByRole("heading", { name: "Thank you!" })
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      message: "Idea",
      share_email: "ada@example.com",
    })
  })

  it("shows the gateway's reason for a refused send, keeping the draft", async () => {
    const { fetch, user } = setup({ email: "ada@example.com" })
    fetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          detail:
            "You're signed in as someone else now. Reload the page and try again.",
        }),
        { status: 422, headers: { "Content-Type": "application/json" } },
      ),
    )
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "You're signed in as someone else now. Reload the page and try again.",
    )
    expect(field()).toHaveValue("Idea")
  })

  it("sends once when Send is pressed twice before the first send renders", async () => {
    const { fetch, user } = setup({ email: "ada@example.com" })
    await user.type(field(), "Idea")
    const send = screen.getByRole("button", { name: SEND })
    // Both presses in one act, as a double click lands before React re-renders
    // with the first send pending.
    act(() => {
      send.click()
      send.click()
    })
    await screen.findByRole("heading", { name: "Thank you!" })
    expect(fetch).toHaveBeenCalledOnce()
  })

  it("holds the compose's height on the floating frame only, never on the sheet", async () => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(320)
    const { user } = setup({ email: "ada@example.com" })
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    await screen.findByRole("heading", { name: "Thank you!" })
    const thanks = screen
      .getByRole("dialog", { name: "Thank you!" })
      .querySelector<HTMLElement>('[data-feedback-view="sent"]')
    expect(thanks?.style.getPropertyValue("--feedback-held-height")).toBe(
      "320px",
    )
    // A pixel floor on the sheet would outlast a rotation to a shorter viewport.
    expect(thanks?.style.minHeight).toBe("")
    expect(thanks).toHaveClass(
      "[@media(width>=640px)_and_(height>=640px)]:min-h-[min(var(--feedback-held-height),calc(var(--visual-viewport-height,100dvh)_-_15rem))]",
    )
  })

  it("sends again after a failed send", async () => {
    const { fetch, user } = setup()
    fetch.mockResolvedValueOnce(new Response(null, { status: 503 }))
    await user.type(field(), "Idea")
    await user.click(screen.getByRole("button", { name: SEND }))
    await screen.findByRole("alert")
    await user.click(screen.getByRole("button", { name: SEND }))
    await screen.findByRole("heading", { name: "Thank you!" })
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it("caps the field at the gateway's 4,000", () => {
    setup()
    expect(field()).toHaveAttribute("maxlength", "4000")
  })
})
