import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { describe, expect, it, vi } from "vitest"

import { DocsPage, markdownComponents } from "@/features/docs/DocsPage"

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
      screen.getByText(/Otari ships with a web admin dashboard for operators/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "The two-key model" }),
    ).toBeInTheDocument()
  })

  it("omits the circular first-run walkthrough but keeps the reference sections", () => {
    render(<DocsPage />)

    // The reader is already past first run (running, signed-in dashboard), so
    // the getting-started walkthrough is trimmed from the in-app view.
    expect(
      screen.queryByRole("heading", { name: "First-run walkthrough" }),
    ).toBeNull()
    expect(screen.queryByText(/Find your master key/)).toBeNull()
    // Sections on both sides of the dropped one still render.
    expect(
      screen.getByRole("heading", { name: "The two-key model" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Page-by-page reference" }),
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
      "https://github.com/mozilla-ai/otari/blob/main/docs/configuration.md",
    )
    expect(configLink).toHaveAttribute("target", "_blank")
    expect(configLink).toHaveAttribute("rel", "noreferrer")
  })
})

/**
 * The code block's label row, covered here rather than by a page measurement,
 * and the reason is worth recording: the only fenced blocks in
 * `docs/dashboard.md` sit in the first-run walkthrough, which this page
 * deliberately drops (a reader who reached it is already signed in). The guide
 * as rendered has 72 inline `<code>` elements and no `<pre>` at all, so the row
 * cannot be seen on the running page however far you scroll. What is asserted
 * is the real pipeline: the page's own component map, driven by ReactMarkdown
 * over a fence.
 */
describe("DocsPage code blocks", () => {
  const renderFence = (md: string) =>
    render(
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={markdownComponents}
      >
        {md}
      </ReactMarkdown>,
    )

  it("labels a fence with its language and offers a copy control", () => {
    renderFence("```bash\nuv run otari serve\n```\n")
    expect(screen.getByText("bash")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument()
    // The block is the scrollable region, named for AT by its language.
    expect(
      screen.getByRole("region", { name: "bash code" }),
    ).toBeInTheDocument()
  })

  it("falls back to a neutral label when the fence names no language", () => {
    // An unlabeled row would be a bar with a control and no subject; "code" is
    // the honest name for a block whose author did not say what it is.
    renderFence("```\nplain\n```\n")
    expect(screen.getByText("code")).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "Code" })).toBeInTheDocument()
  })

  it("copies the block's text and says so", async () => {
    // `userEvent.setup()` installs its own `navigator.clipboard`, so the spy
    // goes on after setup rather than before it, or it is the one that gets
    // replaced and the call lands somewhere nothing is watching.
    const user = userEvent.setup()
    const writeText = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined)
    renderFence("```text\nYour master key: otari-mk-…\n```\n")

    await user.click(screen.getByRole("button", { name: "Copy" }))

    expect(writeText).toHaveBeenCalledWith("Your master key: otari-mk-…\n")
    expect(
      await screen.findByRole("button", { name: "Copied" }),
    ).toBeInTheDocument()
    writeText.mockRestore()
  })

  it("renders no copy control when there is nothing to copy", () => {
    // An empty fence still produces a block; a control that would put an empty
    // string on the clipboard is worse than no control.
    renderFence("```js\n```\n")
    expect(screen.queryByRole("button", { name: "Copy" })).toBeNull()
  })

  it("still renders no fence on the guide itself, loudly", () => {
    // Pinned so this stops being true noisily rather than quietly: if the guide
    // grows a fence outside the dropped walkthrough, the row becomes visible on
    // the page and should be measured there instead of only here.
    const { container } = render(<DocsPage />)
    expect(container.querySelectorAll("pre")).toHaveLength(0)
    expect(container.querySelectorAll("code").length).toBeGreaterThan(0)
  })
})
