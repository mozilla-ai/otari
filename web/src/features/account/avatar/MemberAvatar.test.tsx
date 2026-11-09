import { render } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { saveAvatarChoice } from "./avatarChoice"
import { defaultFaceFor } from "./faces"
import { MemberAvatar } from "./MemberAvatar"

const ADA = "33333333-3333-3333-3333-333333333333"

afterEach(() => {
  window.localStorage.clear()
})

function drawnFace(container: HTMLElement): number | null {
  const svg = container.querySelector("[data-face]")
  return svg ? Number(svg.getAttribute("data-face")) : null
}

describe("MemberAvatar", () => {
  it("holds the row's place with an empty square before there is a user", () => {
    const { container } = render(<MemberAvatar userId={null} />)
    expect(drawnFace(container)).toBeNull()
    expect(container.querySelector("span[aria-hidden]")).toBeInTheDocument()
  })

  it("draws the face the user id gives", () => {
    const { container } = render(<MemberAvatar userId={ADA} />)
    expect(drawnFace(container)).toBe(defaultFaceFor(ADA))
  })

  it("draws a face picked in this browser", () => {
    saveAvatarChoice(ADA, 42)
    const { container } = render(<MemberAvatar userId={ADA} />)
    expect(drawnFace(container)).toBe(42)
  })
})
