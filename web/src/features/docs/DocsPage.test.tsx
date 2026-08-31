import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { DocsPage } from "@/features/docs/DocsPage"

describe("DocsPage", () => {
  it("renders the bundled dashboard guide, not a link to a separate docs site", () => {
    render(<DocsPage />)

    // The page chrome names it as the guide, and the guide content is rendered
    // inline from the bundled Markdown (docs/dashboard.md), so it is
    // discoverable without a docs site.
    expect(
      screen.getByRole("heading", { level: 1, name: "User guide" }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Otari serves its dashboard at the gateway root/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Sign-in and secrets" }),
    ).toBeInTheDocument()
  })

  it("omits the circular first-run walkthrough but keeps the operating sections", () => {
    render(<DocsPage />)

    // The reader is already past first run (running, signed-in dashboard), so
    // the getting-started walkthrough is trimmed from the in-app view.
    expect(
      screen.queryByRole("heading", { name: "First-run walkthrough" }),
    ).toBeNull()
    expect(screen.queryByText(/Start Otari in standalone mode/)).toBeNull()
    // The guide still covers setup and ongoing operation after the dropped section.
    expect(
      screen.getByRole("heading", { name: "The setup guide" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Navigation" }),
    ).toBeInTheDocument()
  })

  it("shows a single top-level heading, dropping the guide's duplicate title", () => {
    render(<DocsPage />)

    // The guide's own "# Admin dashboard" title is stripped so it does not
    // stack a second big heading under the page's "User guide" header.
    const h1s = screen.getAllByRole("heading", { level: 1 })
    expect(h1s).toHaveLength(1)
    expect(h1s[0]).toHaveTextContent("User guide")
  })

  it("renders GFM tables from the guide, keeping table semantics inside a focusable scroll region", () => {
    render(<DocsPage />)

    // The two-key model is a Markdown table; rendering it as a real <table>
    // proves remark-gfm is wired up (plain Markdown would leave it as text).
    const tables = screen.getAllByRole("table")
    expect(tables.length).toBeGreaterThan(0)
    // The horizontal scroll lives on a focusable wrapper so keyboard users can
    // reach clipped columns, and the <table> keeps its native semantics rather
    // than a display:block that would strip role=table in WebKit.
    const region = screen.getAllByRole("region", { name: "Table" })[0]
    expect(region).toHaveAttribute("tabindex", "0")
    expect(region).toContainElement(tables[0])
  })

  it("does not leak react-markdown's node prop onto rendered DOM elements", () => {
    const { container } = render(<DocsPage />)

    // react-markdown passes each hast node to custom components; if it is not
    // destructured out of the DOM spread it renders as node="[object Object]".
    const anchors = container.querySelectorAll("a")
    expect(anchors.length).toBeGreaterThan(0)
    for (const anchor of anchors) {
      expect(anchor.hasAttribute("node")).toBe(false)
    }
  })

  it("rewrites sibling doc links to the GitHub source and opens them in a new tab", () => {
    render(<DocsPage />)

    // The guide links to sibling docs (e.g. configuration.md) that are not
    // bundled here, so a relative link cannot resolve inside the SPA. It is
    // rewritten to the rendered source on GitHub and opened in a new tab.
    const [configLink] = screen.getAllByRole("link", { name: /configuration/i })
    expect(configLink).toHaveAttribute(
      "href",
      "https://github.com/mozilla-ai/otari/blob/main/docs/configuration.md#mail",
    )
    expect(configLink).toHaveAttribute("target", "_blank")
    expect(configLink).toHaveAttribute("rel", "noreferrer")
  })
})
