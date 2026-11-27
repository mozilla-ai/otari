import { screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { renderWithRouter } from "@/tests/router"
import { ToolsFrame } from "./ToolsFrame"

describe("ToolsFrame", () => {
  it("titles the list Tools, with no way back", async () => {
    await renderWithRouter(<ToolsFrame>LIST</ToolsFrame>)

    expect(
      await screen.findByRole("heading", { level: 1, name: "Tools" }),
    ).toBeInTheDocument()
    expect(screen.getByText("LIST")).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "← Tools" })).toBeNull()
    expect(screen.queryByRole("navigation")).toBeNull()
  })

  it("titles a tool's settings by the tool and links back to the list above the title", async () => {
    await renderWithRouter(
      <ToolsFrame title="Web search" back>
        SETTINGS
      </ToolsFrame>,
    )

    const title = await screen.findByRole("heading", {
      level: 1,
      name: "Web search",
    })
    expect(screen.queryByRole("heading", { name: "Tools" })).toBeNull()
    const back = screen.getByRole("link", { name: "← Tools" })
    expect(back).toHaveAttribute("href", "/tools")
    expect(
      back.compareDocumentPosition(title) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(screen.getByRole("navigation", { name: "Back" })).toBeInTheDocument()
  })
})
