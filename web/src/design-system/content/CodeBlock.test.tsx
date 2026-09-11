import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { CodeBlock } from "./CodeBlock"

describe("CodeBlock", () => {
  it("labels the row with its language and names the block for AT", () => {
    render(<CodeBlock label="bash" value="uv run otari serve" />)

    expect(screen.getByText("bash")).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "bash code" })).toHaveTextContent(
      "uv run otari serve",
    )
  })

  it("falls back to a neutral label when it is given no language", () => {
    // An unlabeled row would be a bar with a control and no subject.
    render(<CodeBlock value="plain" />)

    expect(screen.getByText("code")).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "Code" })).toBeInTheDocument()
  })

  it("offers no copy control when there is nothing to copy", () => {
    // Absent rather than dead: a control that copies an empty string teaches
    // nothing about why it did.
    render(<CodeBlock label="text">a rendered form only</CodeBlock>)

    expect(screen.queryByRole("button", { name: /^Copy / })).toBeNull()
  })

  it("copies the value rather than what is rendered, and says so", async () => {
    // The two differ on purpose: the bundled guide renders react-markdown's own
    // nodes while the clipboard gets the raw fence, and the setup guide renders
    // a concealed key while the clipboard gets the real one.
    const user = userEvent.setup()
    const writeText = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined)
    render(
      <CodeBlock label="curl" value="curl --header 'Otari-Key: gw-real'">
        curl --header &apos;Otari-Key: ••••&apos;
      </CodeBlock>,
    )

    await user.click(screen.getByRole("button", { name: "Copy curl" }))

    expect(writeText).toHaveBeenCalledWith("curl --header 'Otari-Key: gw-real'")
    expect(
      await screen.findByRole("button", { name: "Copied" }),
    ).toBeInTheDocument()
  })

  it("never claims a copy the clipboard refused", async () => {
    // This dashboard is routinely served from a non-secure origin, and the
    // shared helper's `execCommand` fallback is not available in jsdom either.
    const user = userEvent.setup()
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("denied"),
    )
    render(<CodeBlock label="curl" value="curl https://example.com" />)

    await user.click(screen.getByRole("button", { name: "Copy curl" }))

    expect(
      screen.getByRole("button", { name: "Copy curl" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Copied" })).toBeNull()
  })
})
