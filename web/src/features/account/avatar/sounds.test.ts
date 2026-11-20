import { play } from "cuelume"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { playPickerSound } from "./sounds"

vi.mock("cuelume", () => ({ play: vi.fn(), unlock: vi.fn() }))

describe("playPickerSound", () => {
  let now = 10_000
  beforeEach(() => {
    vi.mocked(play).mockClear()
    now += 10_000
    vi.spyOn(performance, "now").mockImplementation(() => now)
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("plays one tick per 45ms however fast faces pass", () => {
    playPickerSound("pass", "forward")
    now += 20
    playPickerSound("pass", "forward")
    now += 30
    playPickerSound("pass", "forward")

    expect(play).toHaveBeenCalledTimes(2)
    expect(play).toHaveBeenLastCalledWith("select", {
      theme: "bubble",
      emphasis: "subtle",
      volume: 0.8,
      direction: "forward",
    })
  })

  it("never drops the landing, even right after a tick", () => {
    playPickerSound("pass", "back")
    now += 5
    playPickerSound("land")

    expect(vi.mocked(play).mock.calls.map(([cue]) => String(cue))).toEqual([
      "select",
      "select",
    ])
  })
})
