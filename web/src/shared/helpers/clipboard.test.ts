import { afterEach, describe, expect, it, vi } from "vitest"

import { copyToClipboard } from "@/shared/helpers/clipboard"

// jsdom implements no clipboard at all: neither navigator.clipboard nor
// document.execCommand exists, so each path is installed per test.
function stubExecCommand(
  result: boolean | (() => never),
): ReturnType<typeof vi.fn> {
  const execCommand = vi.fn(
    typeof result === "function" ? result : () => result,
  )
  Object.defineProperty(document, "execCommand", {
    value: execCommand,
    configurable: true,
    writable: true,
  })
  return execCommand
}

describe("copyToClipboard", () => {
  afterEach(() => {
    Reflect.deleteProperty(document, "execCommand")
    vi.restoreAllMocks()
  })

  it("reports copying only after the clipboard write succeeds", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    const copied = await copyToClipboard("provider exploded", { writeText })
    expect(writeText).toHaveBeenCalledWith("provider exploded")
    expect(copied).toBe(true)
  })

  it("falls back to execCommand when the Clipboard API write is refused", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("clipboard denied"))
    const execCommand = stubExecCommand(true)

    expect(await copyToClipboard("openai:gpt-4o", { writeText })).toBe(true)
    expect(execCommand).toHaveBeenCalledWith("copy")
  })

  it("copies on a non-secure origin, where there is no Clipboard API at all", async () => {
    // The dashboard is routinely served over plain HTTP on a LAN address, where
    // navigator.clipboard is undefined; the copy still has to work (#478).
    const execCommand = stubExecCommand(true)

    expect(
      await copyToClipboard("anthropic:claude-opus-4-5-20251101", undefined),
    ).toBe(true)
    expect(execCommand).toHaveBeenCalledWith("copy")
  })

  it("leaves no scratch textarea behind, and restores the operator's selection", async () => {
    stubExecCommand(true)
    const paragraph = document.createElement("p")
    paragraph.textContent = "already selected"
    document.body.appendChild(paragraph)
    const range = document.createRange()
    range.selectNodeContents(paragraph)
    document.getSelection()?.addRange(range)

    await copyToClipboard("openai:gpt-4o-mini", undefined)

    expect(document.querySelectorAll("textarea")).toHaveLength(0)
    expect(document.getSelection()?.toString()).toBe("already selected")
    paragraph.remove()
  })

  it("puts focus back where it was, so a keyboard copy keeps its tab position", async () => {
    // The scratch textarea's select() takes focus in a real browser, so removing
    // it drops focus to <body>. jsdom does not move focus on select(), so this
    // pins the restore call rather than reproducing the loss.
    stubExecCommand(true)
    const button = document.createElement("button")
    document.body.appendChild(button)
    button.focus()

    await copyToClipboard("openai:gpt-4o", undefined)

    expect(document.activeElement).toBe(button)
    button.remove()
  })

  it("does not report success when both paths fail", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("clipboard denied"))
    stubExecCommand(false)
    expect(await copyToClipboard("provider exploded", { writeText })).toBe(
      false,
    )
  })

  it("does not report success when execCommand throws", async () => {
    stubExecCommand(() => {
      throw new Error("not allowed")
    })
    expect(await copyToClipboard("openai:gpt-4o", undefined)).toBe(false)
  })

  it("does not report success where no clipboard mechanism exists at all", async () => {
    expect(await copyToClipboard("openai:gpt-4o", undefined)).toBe(false)
  })
})
