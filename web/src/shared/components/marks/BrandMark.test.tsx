import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import {
  anyMakerMark,
  anyProviderMark,
  MakerMark,
  ProviderMark,
} from "@/shared/components/marks/BrandMark"

describe("ProviderMark", () => {
  it("draws the vendor's mark when there is one", () => {
    const { container } = render(<ProviderMark providerId="mistral" />)

    const svg = container.querySelector("svg")
    expect(svg).not.toBeNull()
    expect(svg?.getAttribute("viewBox")).toBe("0 0 24 24")
    expect(container.querySelectorAll("path").length).toBeGreaterThan(0)
  })

  it("inherits the page's text ink rather than carrying a color", () => {
    const { container } = render(<ProviderMark providerId="openai" />)

    // `fill-current` is the whole colour story: no fill attribute on the paths,
    // nothing per-vendor, so both themes get an ink that contrasts.
    expect(container.querySelector("svg")?.getAttribute("class")).toContain(
      "fill-current",
    )
    expect(container.querySelector("path")?.getAttribute("fill")).toBeNull()
  })

  it("hides the mark from assistive tech, because the name is beside it", () => {
    const { container } = render(<ProviderMark providerId="anthropic" />)

    expect(container.querySelector("svg")?.getAttribute("aria-hidden")).toBe(
      "true",
    )
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

  it("draws the smaller step when asked", () => {
    const { container } = render(
      <ProviderMark providerId="mistral" step={14} />,
    )

    expect(container.querySelector("svg")?.getAttribute("class")).toContain(
      "size-3.5",
    )
  })

  it("gives our own provider ids the Otari mark, not a tile", () => {
    // Both ids this deployment serves its own models under. Showing ourselves a
    // lettermark while every third party got a logo is the bug this feature
    // exists to fix, pointed at us.
    for (const id of ["mzai", "otari"]) {
      const { container } = render(<ProviderMark providerId={id} />)

      expect(container.querySelector("svg")?.getAttribute("viewBox")).toBe(
        "0 0 273 250",
      )
      expect(container.querySelector("path")?.getAttribute("fill")).toBe(
        "currentColor",
      )
    }
  })

  it("applies the wrapping transform a mark needs", () => {
    // llama.cpp is the one mark whose geometry is not on a 24 grid; its group
    // transform is what puts it in its own viewBox.
    const { container } = render(<ProviderMark providerId="llamacpp" />)

    expect(container.querySelector("svg")?.getAttribute("viewBox")).toBe(
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
  it("draws the maker's mark, keyed on the vendor slug", () => {
    const { container } = render(
      <MakerMark vendorSlug="mistralai" label="Mistral AI" />,
    )

    expect(container.querySelector("svg")).not.toBeNull()
    expect(container.querySelectorAll("path").length).toBeGreaterThan(0)
  })

  it("defaults to the 14px step, which is what the caption lines take", () => {
    const { container } = render(<MakerMark vendorSlug="meta" label="Meta" />)

    expect(container.querySelector("svg")?.getAttribute("class")).toContain(
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
    expect(anyMakerMark(["amazon", "openbmb"])).toBe(false)
    expect(anyMakerMark([])).toBe(false)
  })
})
