import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { defaultFaceFor } from "./faces"
import { MemberAvatar } from "./MemberAvatar"

const ADA = "33333333-3333-3333-3333-333333333333"

function drawnFace(container: HTMLElement): number | null {
  const svg = container.querySelector("[data-face]")
  return svg ? Number(svg.getAttribute("data-face")) : null
}

describe("MemberAvatar", () => {
  it("holds the row's place with an empty square before there is a user", () => {
    const { container } = render(<MemberAvatar userId={null} face={null} />)
    expect(drawnFace(container)).toBeNull()
    expect(container.querySelector("span[aria-hidden]")).toBeInTheDocument()
  })

  it("draws the face the user id gives when they have picked none", () => {
    const { container } = render(<MemberAvatar userId={ADA} face={null} />)
    expect(drawnFace(container)).toBe(defaultFaceFor(ADA))
  })

  it("draws the face the member picked, from their member row", () => {
    const { container } = render(<MemberAvatar userId={ADA} face={42} />)
    expect(drawnFace(container)).toBe(42)
  })

  it("ignores a pick left in this browser, which only its owner's row reads", () => {
    window.localStorage.setItem(`otari.avatar:${ADA}`, "42")
    const { container } = render(<MemberAvatar userId={ADA} face={null} />)
    expect(drawnFace(container)).toBe(defaultFaceFor(ADA))
    window.localStorage.clear()
  })
})
