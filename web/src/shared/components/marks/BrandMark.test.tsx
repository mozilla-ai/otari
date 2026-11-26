import { render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import {
  anyHarnessMark,
  anyMakerMark,
  anyProviderMark,
  HarnessMark,
  MakerMark,
  ProviderMark,
} from "@/shared/components/marks/BrandMark"
import { languageMark } from "@/shared/helpers/brandMarks"

/**
 * The rendered `<svg>`, once the lazy geometry chunk has resolved.
 *
 * `BrandMark` defers the path data, so the first paint is the reserved box and
 * the mark appears a microtask later. Every assertion about the drawing has to
 * wait for it; the tile cases do not, since the tile is not deferred.
 */
async function glyphOf(container: HTMLElement): Promise<SVGElement> {
  return await waitFor(() => {
    const svg = container.querySelector("svg")
    if (svg === null) throw new Error("no svg yet")
    return svg
  })
}

describe("ProviderMark", () => {
  it("draws the vendor's mark when there is one", async () => {
    const { container } = render(<ProviderMark providerId="mistral" />)

    const svg = await glyphOf(container)
    expect(svg.getAttribute("viewBox")).toBe("0 0 24 24")
    expect(container.querySelectorAll("path").length).toBeGreaterThan(0)
  })

  it("inherits the page's text ink rather than carrying a color", async () => {
    const { container } = render(<ProviderMark providerId="openai" />)

    // `fill-current` is the whole color story: no fill attribute on the paths,
    // nothing per-vendor, so both themes get an ink that contrasts.
    const svg = await glyphOf(container)
    expect(svg.getAttribute("class")).toContain("fill-current")
    expect(container.querySelector("path")?.getAttribute("fill")).toBeNull()
  })

  it("hides the mark from assistive tech, because the name is beside it", async () => {
    const { container } = render(<ProviderMark providerId="anthropic" />)

    expect((await glyphOf(container)).getAttribute("aria-hidden")).toBe("true")
  })

  it("falls back to a lettermark tile for a provider with no mark", () => {
    render(<ProviderMark providerId="my-local-llm" />)

    // The tile, not a blank and not a broken image.
    expect(screen.getByText("m")).toBeInTheDocument()
  })

  it("takes the tile's initial from the displayed label, not the id", () => {
    // A renamed instance tiles as the operator spelled it. Keyed on the label
    // because that is what the reader sees on the row.
    render(<ProviderMark providerId="acme-7" label="Zephyr Labs" />)

    expect(screen.getByText("Z")).toBeInTheDocument()
  })

  it("names the provider on the tile when no label is given", () => {
    // Falls through the display-name map, so a known id still tiles as its
    // brand rather than as its wire id: "Eden AI", not "edenai".
    render(<ProviderMark providerId="edenai" />)

    expect(screen.getByText("E")).toBeInTheDocument()
  })

  it("draws the smaller step when asked", async () => {
    const { container } = render(
      <ProviderMark providerId="mistral" step={14} />,
    )

    expect((await glyphOf(container)).getAttribute("class")).toContain(
      "size-3.5",
    )
  })

  it("gives our own provider ids the Otari mark, not a tile", () => {
    // The ids this deployment serves its own models under, and the platform
    // instance. Showing ourselves a
    // lettermark while every third party got a logo is the bug this feature
    // exists to fix, pointed at us.
    for (const id of ["mzai", "otari", "otari-ai"]) {
      const { container } = render(<ProviderMark providerId={id} />)

      expect(container.querySelector("svg")?.getAttribute("viewBox")).toBe(
        "0 0 273 250",
      )
      expect(container.querySelector("path")?.getAttribute("fill")).toBe(
        "currentColor",
      )
    }
  })

  it("applies the wrapping transform a mark needs", async () => {
    // llama.cpp is the one mark whose geometry is not on a 24 grid; its group
    // transform is what puts it in its own viewBox.
    const { container } = render(<ProviderMark providerId="llamacpp" />)

    expect((await glyphOf(container)).getAttribute("viewBox")).toBe(
      "0 0 250 250",
    )
    expect(container.querySelector("g")?.getAttribute("transform")).toBeTruthy()
  })
})

describe("anyProviderMark", () => {
  it("is true when at least one provider resolves", () => {
    expect(anyProviderMark(["my-local-llm", "mistral"])).toBe(true)
  })

  it("is false when none does, so the list reserves no slot", () => {
    expect(anyProviderMark(["my-local-llm", "staging-box"])).toBe(false)
    expect(anyProviderMark([])).toBe(false)
  })
})

describe("MakerMark", () => {
  it("draws the maker's mark, keyed on the vendor slug", async () => {
    const { container } = render(
      <MakerMark vendorSlug="mistralai" label="Mistral AI" />,
    )

    await glyphOf(container)
    expect(container.querySelectorAll("path").length).toBeGreaterThan(0)
  })

  it("defaults to the 14px step, which is what the caption lines take", async () => {
    const { container } = render(<MakerMark vendorSlug="meta" label="Meta" />)

    expect((await glyphOf(container)).getAttribute("class")).toContain(
      "size-3.5",
    )
  })

  it("tiles a maker with no mark, under its own initial", () => {
    // Amazon by ruling, since its only mark is the aws wordmark.
    render(<MakerMark vendorSlug="amazon" label="Amazon" />)

    expect(screen.getByText("A")).toBeInTheDocument()
  })

  it("does not resolve a provider id as a maker", () => {
    // The two key spaces are separate on purpose; `mistral` is the provider.
    render(<MakerMark vendorSlug="mistral" label="Mistral AI" />)

    expect(screen.getByText("M")).toBeInTheDocument()
  })
})

describe("anyMakerMark", () => {
  it("is true when at least one maker resolves", () => {
    expect(anyMakerMark(["amazon", "mistralai"])).toBe(true)
  })

  it("is false when none does", () => {
    expect(anyMakerMark(["amazon", "writer"])).toBe(false)
    expect(anyMakerMark([])).toBe(false)
  })
})

describe("HarnessMark", () => {
  it("draws an SDK's language, since it has no mark of its own", async () => {
    const { container } = render(
      <HarnessMark harness="openai" language="python" />,
    )

    await glyphOf(container)
    expect(container.querySelector("path")?.getAttribute("d")).toBe(
      languageMark("python")?.shapes[0].d,
    )
  })

  it("draws the language of a session that names no harness", async () => {
    const { container } = render(<HarnessMark harness={null} language="go" />)

    await glyphOf(container)
    expect(anyHarnessMark([{ harness: null, client_language: "go" }])).toBe(
      true,
    )
  })

  it("tiles a client with no mark and no language", () => {
    render(<HarnessMark harness="my-agent" />)

    expect(screen.getByText("m")).toBeInTheDocument()
  })

  it("keeps an empty, silent slot for a session with no client", () => {
    // The slot keeps the session's name aligned with the rows that have a mark.
    const { container } = render(<HarnessMark harness={null} />)

    const slot = container.firstElementChild
    expect(slot?.tagName).toBe("SPAN")
    expect(slot?.getAttribute("aria-hidden")).toBe("true")
    expect(slot?.className).toContain("size-4")
    expect(slot?.className).toContain("shrink-0")
    expect(slot?.childElementCount).toBe(0)
    expect(container.textContent).toBe("")
  })

  it("names the language for a screen reader, since the mark is decorative", () => {
    render(<HarnessMark harness={null} language="go" />)

    expect(screen.getByText("Client language: Go")).toHaveClass("sr-only")
  })
})
