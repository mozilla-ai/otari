import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  CONCEALED_SECRET,
  CopyableValue,
  CopyField,
} from "@/shared/components/actions/CopyField"

describe("CopyField", () => {
  // Scoped like CopyButton's above: the failure case spies on
  // `navigator.clipboard`, and without this it would leak into the describes
  // that follow.
  afterEach(() => {
    vi.restoreAllMocks()
  })

  // Written against the arrangement that shipped before the in-field control
  // existed, so it is a regression guard for the six call sites that keep it
  // rather than a description of the newer branch.
  it("keeps the Copy button in the label row by default", () => {
    render(<CopyField label="Secret key" value="sk-test-000" />)

    const field = screen.getByLabelText("Secret key")
    expect(field.tagName).toBe("INPUT")
    expect(field).toHaveAttribute("readonly")
    // The default arrangement's button is the field's sibling in the label row,
    // named by its own text, and there is no control inside the field.
    expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Copy Secret key" }),
    ).not.toBeInTheDocument()
  })

  it("renders the multiline variant as a textarea", () => {
    render(<CopyField label="curl" value={"line one\nline two"} multiline />)

    const field = screen.getByLabelText("curl")
    expect(field.tagName).toBe("TEXTAREA")
    expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument()
  })

  it("moves the copy affordance into the field when given an action", () => {
    render(
      <CopyField
        label="TXT record for example.com"
        value="otari-verify=abc"
        action={<button type="button">Verify domain</button>}
      />,
    )

    const field = screen.getByLabelText("TXT record for example.com")
    expect(field.tagName).toBe("INPUT")
    expect(field).toHaveAttribute("readonly")
    // The label row's button is gone, and the control inside the field is named
    // by the label rather than by bare text, so a screen reader hears which
    // field it copies.
    expect(
      screen.queryByRole("button", { name: "Copy" }),
    ).not.toBeInTheDocument()
    const copy = screen.getByRole("button", {
      name: "Copy TXT record for example.com",
    })
    // The control is inside the field's wrapper; the action is its sibling.
    expect(copy.closest("div")).toContainElement(field)
    expect(
      screen.getByRole("button", { name: "Verify domain" }),
    ).toBeInTheDocument()
  })

  it("keeps the confirmation out of the row, so the action cannot shift", async () => {
    const user = userEvent.setup()
    const { container } = render(
      <CopyField
        label="TXT record for example.com"
        value="otari-verify=abc"
        action={<button type="button">Verify domain</button>}
      />,
    )
    // `sr-only` is clipped and out of flow, so it is not part of what the row
    // lays out. Everything else here would be, including a confirmation line.
    const laidOut = () => {
      const clone = container.cloneNode(true) as HTMLElement
      for (const node of clone.querySelectorAll(".sr-only")) node.remove()
      return clone.textContent
    }
    const before = laidOut()

    await user.click(
      screen.getByRole("button", {
        name: "Copy TXT record for example.com",
      }),
    )
    await screen.findByText("Copied!")

    // The acknowledgement is an overlay, so the Verify button does not move.
    expect(laidOut()).toBe(before)
  })

  it("selects the value when a copy fails, so Ctrl/Cmd-C still works", async () => {
    const user = userEvent.setup()
    // Both clipboard paths refused, which is the case this fallback exists for.
    // The async API is made to throw; the legacy path then fails on its own,
    // because jsdom defines no `document.execCommand` and `legacyCopy` reports
    // false when the call throws. That is what the existing "announces a
    // blocked copy" case above relies on too.
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("not a secure context"),
    )
    render(
      <CopyField
        label="TXT record for example.com"
        value="otari-verify=abc"
        action={<button type="button">Verify domain</button>}
      />,
    )
    const field = screen.getByLabelText(
      "TXT record for example.com",
    ) as HTMLInputElement

    await user.click(
      screen.getByRole("button", {
        name: "Copy TXT record for example.com",
      }),
    )

    await waitFor(() => {
      expect(document.activeElement).toBe(field)
      expect(field.selectionStart).toBe(0)
      expect(field.selectionEnd).toBe("otari-verify=abc".length)
    })
  })

  it("selects only after the copy attempt, not before it", async () => {
    const user = userEvent.setup()
    // The order is the thing under test, not just the end state. The legacy
    // fallback restores the selection and focus it found on its way out, so a
    // selection made before the attempt is undone by the fallback itself. That
    // does not reproduce in jsdom (there is no `execCommand` for the fallback
    // to get that far), so the end state alone passes either way and the
    // sequence has to be asserted directly.
    const order: string[] = []
    vi.spyOn(navigator.clipboard, "writeText").mockImplementation(() => {
      order.push("attempt")
      return Promise.reject(new Error("not a secure context"))
    })
    render(
      <CopyField
        label="TXT record for example.com"
        value="otari-verify=abc"
        action={<button type="button">Verify domain</button>}
      />,
    )
    const field = screen.getByLabelText("TXT record for example.com")
    field.addEventListener("focus", () => order.push("focus"))

    await user.click(
      screen.getByRole("button", {
        name: "Copy TXT record for example.com",
      }),
    )

    await waitFor(() => expect(order).toContain("focus"))
    expect(order).toEqual(["attempt", "focus"])
  })

  it("rejects a falsy action at the type level", () => {
    // `ReactNode` admitted `false`, so `action={enabled && <Button />}`
    // compiled and then fell back to the default arrangement through
    // `if (action)`. `ReactElement` rejects it where it is written.
    const enabled = false
    render(
      <CopyField
        label="TXT record"
        value="otari-verify=abc"
        // @ts-expect-error a conditional action is a falsy action, which would
        // silently render the label-row button and the clipboard-only path.
        action={enabled && <button type="button">Verify</button>}
      />,
    )
    expect(screen.getByLabelText("TXT record")).toBeInTheDocument()
  })

  it("rejects an action on the multiline variant at the type level", () => {
    render(
      // @ts-expect-error a textarea's right padding indents every line, so the
      // in-field arrangement is not available to the multiline variant. The
      // union rejects the pair at the element, which is where the error lands.
      <CopyField
        label="curl"
        value="one"
        multiline
        action={<button type="button">Verify domain</button>}
      />,
    )
    expect(screen.getByLabelText("curl")).toBeInTheDocument()
  })
})

describe("CopyField, concealed", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows the stand-in until the value is asked for, and hides it again", async () => {
    const user = userEvent.setup()
    render(
      <CopyField
        label="Secret key"
        value="gw-real-secret"
        concealed={CONCEALED_SECRET}
      />,
    )

    const field = screen.getByLabelText("Secret key")
    expect(field).toHaveValue(CONCEALED_SECRET)
    // The plaintext is not in the document at all, so it cannot be read off
    // the screen, scraped out of a screenshot, or found in a DOM dump.
    expect(document.body.textContent).not.toContain("gw-real-secret")

    await user.click(screen.getByRole("button", { name: "Show Secret key" }))
    expect(field).toHaveValue("gw-real-secret")

    await user.click(screen.getByRole("button", { name: "Hide Secret key" }))
    expect(field).toHaveValue(CONCEALED_SECRET)
  })

  it("copies the real value while it is concealed", async () => {
    const user = userEvent.setup()
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    })
    render(
      <CopyField
        label="Secret key"
        value="gw-real-secret"
        concealed={CONCEALED_SECRET}
      />,
    )

    await user.click(screen.getByRole("button", { name: "Copy" }))

    // Handed over without being seen, which is the whole point: the clipboard
    // gets the key and the field still shows the stand-in.
    expect(writeText).toHaveBeenCalledWith("gw-real-secret")
    expect(screen.getByLabelText("Secret key")).toHaveValue(CONCEALED_SECRET)
    expect(await screen.findByText("Copied to clipboard.")).toBeInTheDocument()
  })

  it("reveals and selects when no clipboard path will take it", async () => {
    const user = userEvent.setup()
    // Both paths refused, as on a plain-HTTP origin whose browser also has no
    // `execCommand`: the async API is made to throw and jsdom defines no
    // legacy fallback, so nothing wrote to the clipboard.
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("not a secure context"),
    )
    render(
      <CopyField
        label="Secret key"
        value="gw-real-secret"
        concealed={CONCEALED_SECRET}
      />,
    )
    const field = screen.getByLabelText("Secret key") as HTMLInputElement

    await user.click(screen.getByRole("button", { name: "Copy" }))

    // Ctrl/Cmd-C is the only way left, and it can only reach the plaintext, so
    // the field reveals it and selects that rather than the stand-in.
    await waitFor(() => {
      expect(field).toHaveValue("gw-real-secret")
      expect(document.activeElement).toBe(field)
      expect(field.selectionStart).toBe(0)
      expect(field.selectionEnd).toBe("gw-real-secret".length)
    })
    expect(
      screen.getByText("Revealed and selected. Press Ctrl/Cmd-C to copy."),
    ).toBeInTheDocument()
    // Never a claim it copied, which is the rule the whole component follows.
    expect(screen.queryByText("Copied to clipboard.")).not.toBeInTheDocument()
  })

  it("leaves focus unselected while concealed, so no Ctrl/Cmd-C copies bullets", async () => {
    const user = userEvent.setup()
    render(
      <CopyField
        label="Secret key"
        value="gw-real-secret"
        concealed={CONCEALED_SECRET}
      />,
    )
    const field = screen.getByLabelText("Secret key") as HTMLInputElement

    await user.click(field)
    expect(field.selectionStart).toBe(field.selectionEnd)

    // Revealed, the field selects on focus the way every other one does.
    await user.click(screen.getByRole("button", { name: "Show Secret key" }))
    await user.click(field)
    expect(field.selectionStart).toBe(0)
    expect(field.selectionEnd).toBe("gw-real-secret".length)
  })

  it("conceals a snippet around the stand-in, with the toggle out of the field", async () => {
    const user = userEvent.setup()
    const snippet = `curl https://otari.test \\\n  -H "Otari-Key: gw-real-secret"`
    render(
      <CopyField
        label="curl"
        value={snippet}
        concealed={snippet.replace("gw-real-secret", CONCEALED_SECRET)}
        multiline
      />,
    )

    const field = screen.getByLabelText("curl")
    // The request is readable while the key inside it is not: the address is
    // what the operator has to check, and the key is not.
    expect(field).toHaveTextContent("https://otari.test")
    expect((field as HTMLTextAreaElement).value).not.toContain("gw-real-secret")

    await user.click(screen.getByRole("button", { name: "Show curl" }))
    expect(field).toHaveValue(snippet)
  })

  it("conceals a second value, rather than inheriting the first one's reveal", async () => {
    const user = userEvent.setup()
    const { rerender } = render(
      <CopyField
        label="Secret key"
        value="gw-first-secret"
        concealed={CONCEALED_SECRET}
      />,
    )

    await user.click(screen.getByRole("button", { name: "Show Secret key" }))
    expect(screen.getByLabelText("Secret key")).toHaveValue("gw-first-secret")

    // A rotation with the previous key still on screen: the replacement is a
    // credential nobody has asked to see yet.
    rerender(
      <CopyField
        label="Secret key"
        value="gw-second-secret"
        concealed={CONCEALED_SECRET}
      />,
    )

    expect(screen.getByLabelText("Secret key")).toHaveValue(CONCEALED_SECRET)
    expect(document.body.textContent).not.toContain("gw-second-secret")
  })

  it("does not reveal a credential that arrived while a copy was failing", async () => {
    const user = userEvent.setup()
    // A copy left in flight, so the swap lands between the attempt and its
    // failure. Both clipboard paths refuse, which is the branch that reveals.
    let refuse: (reason: Error) => void = () => {}
    vi.spyOn(navigator.clipboard, "writeText").mockReturnValue(
      new Promise((_resolve, reject) => {
        refuse = reject
      }),
    )
    const { rerender } = render(
      <CopyField
        label="Secret key"
        value="gw-first-secret"
        concealed={CONCEALED_SECRET}
      />,
    )

    await user.click(screen.getByRole("button", { name: "Copy" }))
    rerender(
      <CopyField
        label="Secret key"
        value="gw-second-secret"
        concealed={CONCEALED_SECRET}
      />,
    )
    refuse(new Error("not a secure context"))

    // The failure belongs to the key that is gone, so it reveals nothing and
    // claims nothing: the replacement is a credential nobody has asked to see.
    await waitFor(() => {
      expect(screen.getByLabelText("Secret key")).toHaveValue(CONCEALED_SECRET)
    })
    expect(document.body.textContent).not.toContain("gw-second-secret")
    expect(
      screen.queryByText("Revealed and selected. Press Ctrl/Cmd-C to copy."),
    ).not.toBeInTheDocument()
  })

  it("rejects concealing a field that carries an action, at the type level", () => {
    render(
      // @ts-expect-error nothing hands out a credential beside a
      // domain-verification action, and the in-field arrangement has no room
      // for a second control.
      <CopyField
        label="TXT record"
        value="otari-verify=abc"
        concealed={CONCEALED_SECRET}
        action={<button type="button">Verify domain</button>}
      />,
    )
    expect(screen.getByLabelText("TXT record")).toBeInTheDocument()
  })
})

describe("CopyableValue", () => {
  it("copies the value, which need not be what is displayed", async () => {
    const user = userEvent.setup()
    render(
      <CopyableValue value="openai:gpt-4o-2024-11-20" label="model id">
        gpt-4o-2024-11-20
      </CopyableValue>,
    )

    expect(screen.getByText("gpt-4o-2024-11-20")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Copy model id" }))
    expect(await navigator.clipboard.readText()).toBe(
      "openai:gpt-4o-2024-11-20",
    )
  })

  it("keeps a row press from starting on the value, so a drag can highlight it", () => {
    // The whole reason highlighting an id in a table used to fail: react-aria's
    // row press toggles selection on pointer down, and that re-render lands
    // mid-drag and discards the browser's nascent selection (#478). The value
    // stops the pointer sequence from reaching the row.
    const onRowPointerDown = vi.fn()
    const onRowMouseDown = vi.fn()
    render(
      // biome-ignore lint/a11y/noStaticElementInteractions: a stand-in for the table row whose handlers this test proves are not reached
      <div onPointerDown={onRowPointerDown} onMouseDown={onRowMouseDown}>
        <CopyableValue
          value="anthropic:claude-opus-4-5-20251101"
          label="model id"
        />
      </div>,
    )

    const value = screen.getByText("anthropic:claude-opus-4-5-20251101")
    fireEvent.pointerDown(value, {
      pointerId: 1,
      pointerType: "mouse",
      button: 0,
    })
    fireEvent.mouseDown(value, { button: 0 })

    expect(onRowPointerDown).not.toHaveBeenCalled()
    expect(onRowMouseDown).not.toHaveBeenCalled()
    // Selectable in its own right, so an inherited `user-select: none` from a
    // press elsewhere in the row cannot suppress it.
    expect(value.className).toContain("select-text")
  })
})
