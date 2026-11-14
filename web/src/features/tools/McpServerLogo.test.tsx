import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { logoTile, McpServerLogo } from "@/features/tools/McpServerLogo"

describe("logoTile", () => {
  it("tiles a logo only on the theme where it would vanish, on the other theme's surface", () => {
    expect(logoTile("dark", "dark")).toEqual({ tiled: true, surface: "light" })
    expect(logoTile("dark", "light")).toEqual({ tiled: false })
    expect(logoTile("light", "light")).toEqual({ tiled: true, surface: "dark" })
    expect(logoTile("light", "dark")).toEqual({ tiled: false })
    expect(logoTile("tile", "light")).toEqual({
      tiled: true,
      surface: undefined,
    })
    expect(logoTile(null, "dark")).toEqual({ tiled: false })
  })
})

describe("McpServerLogo", () => {
  it("draws the gateway's image, and the first letter when there is none", () => {
    const { container, rerender } = render(
      <McpServerLogo
        name="sentry"
        logoUrl="/api/v1/logo?v=1"
        tone="dark"
        theme="dark"
      />,
    )
    const image = container.querySelector("img")
    expect(image?.getAttribute("src")).toBe("/api/v1/logo?v=1")
    expect(container.querySelector("[data-theme='light']")).not.toBeNull()

    rerender(<McpServerLogo name="sentry" logoUrl={null} />)
    expect(screen.getByText("S")).toBeInTheDocument()
  })
})
