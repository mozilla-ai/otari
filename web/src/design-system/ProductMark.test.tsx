import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ProductMark } from "./ProductMark"

/**
 * The mark, and the three claims its docstring makes that are checkable here.
 *
 * It takes its ink from the text around it, it is decorative so assistive tech
 * never announces it, and it carries no intrinsic size: 273 by 250 is not
 * square, so it is given a width by the caller and left to find its own height.
 * That last one is the claim with a measured failure behind it. A width alone
 * left this rendering 28 by 16 inside a flex row, a ratio of 1.75 against the
 * artwork's 1.09, because an `<svg>` is a replaced element and takes its height
 * from the box rather than from the viewBox.
 *
 * What is asserted is the half this component owns: the viewBox is intact and no
 * width or height attribute is baked in. The other half, that a caller passes
 * `h-auto` alongside its width, is not something this component can enforce, so
 * it is not tested here.
 */
describe("ProductMark", () => {
  const mark = (container: HTMLElement) =>
    container.querySelector("svg") as SVGElement

  it("takes its color from the text around it", () => {
    // `currentColor` rather than a fill of its own. The accent has already moved
    // once in this product's life, and a copy of its value inside an asset is a
    // copy nobody finds when it moves again.
    const { container } = render(<ProductMark />)
    const path = mark(container).querySelector("path") as SVGPathElement
    expect(path).toHaveAttribute("fill", "currentColor")
  })

  it("is decorative, so nothing announces it", () => {
    // The mark sits beside the product name at every call site, so announcing it
    // would read the same word twice.
    const { container } = render(<ProductMark />)
    expect(mark(container)).toHaveAttribute("aria-hidden", "true")
    expect(mark(container)).toHaveAttribute("focusable", "false")
  })

  it("keeps its own aspect and bakes in no size for a caller to fight", () => {
    // 273 by 250, a ratio of 1.09. An intrinsic width or height here would win
    // against the caller's utility and stretch the artwork about 9%, which is
    // the kind of wrong that reads as "something is off" without anybody being
    // able to say what.
    const { container } = render(<ProductMark />)
    expect(mark(container)).toHaveAttribute("viewBox", "0 0 273 250")
    expect(mark(container).hasAttribute("width")).toBe(false)
    expect(mark(container).hasAttribute("height")).toBe(false)
  })

  it("declares its holes to be holes", () => {
    // Intent rather than repair: the shape at the top right is a counter, and a
    // compound path can lose one under the default `nonzero` rule. This artwork
    // does not, because the subpath is wound against the shape around it, so the
    // rule is here to say the holes are holes by construction.
    const { container } = render(<ProductMark />)
    const path = mark(container).querySelector("path") as SVGPathElement
    expect(path).toHaveAttribute("fill-rule", "evenodd")
  })

  it("wears the class the caller sizes it with, and none by default", () => {
    const { container: bare } = render(<ProductMark />)
    expect(mark(bare).getAttribute("class")).toBe("")

    const { container: sized } = render(<ProductMark className="h-auto w-7" />)
    expect(mark(sized)).toHaveClass("h-auto")
    expect(mark(sized)).toHaveClass("w-7")
  })
})
