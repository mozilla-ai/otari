import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { Field } from "@/design-system/forms/Field"

describe("Field", () => {
  it("caps its width so inputs don't stretch across a wide form", () => {
    render(<Field label="API base" value="" onChange={() => {}} />)

    // Bounded rather than filling the whole form width (#328).
    const field = screen.getByRole("textbox").closest(".max-w-md")
    expect(field).toBeInTheDocument()
  })
})

describe("Field autoFocus", () => {
  const realMatchMedia = window.matchMedia
  afterEach(() => {
    window.matchMedia = realMatchMedia
  })

  it("takes focus on mount where the pointer is not coarse", () => {
    render(<Field label="Name" value="" onChange={() => {}} autoFocus />)
    expect(screen.getByRole("textbox")).toHaveFocus()
  })

  it("leaves focus alone on a touch screen, so no keyboard rises by itself", () => {
    window.matchMedia = ((query: string) => ({
      matches: query.includes("coarse"),
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    })) as typeof window.matchMedia
    render(<Field label="Name" value="" onChange={() => {}} autoFocus />)
    expect(screen.getByRole("textbox")).not.toHaveFocus()
  })
})
