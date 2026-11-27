import { screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { renderWithRouter } from "@/tests/router"
import { ToolsFrame } from "./ToolsFrame"

describe("ToolsFrame", () => {
  it("titles the page and leaves a tool's list with no way back", async () => {
    await renderWithRouter(<ToolsFrame>LIST</ToolsFrame>)

    expect(
      await screen.findByRole("heading", { name: "Tools" }),
    ).toBeInTheDocument()
    expect(screen.getByText("LIST")).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "← Tools" })).toBeNull()
    expect(screen.queryByRole("navigation")).toBeNull()
  })

  it("links a tool's settings back to the list", async () => {
    await renderWithRouter(<ToolsFrame back>SETTINGS</ToolsFrame>)

    expect(
      await screen.findByRole("link", { name: "← Tools" }),
    ).toHaveAttribute("href", "/tools")
    expect(screen.queryByRole("navigation")).toBeNull()
  })
})
