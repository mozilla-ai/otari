import { act, render } from "@testing-library/react"
import { useRef } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { LoginBackground } from "./LoginBackground"
import saved from "./login-background.json"
import { drawBars } from "./renderBars"

vi.mock("./renderBars", () => ({ drawBars: vi.fn() }))

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function Harness({
  paused = false,
  config = saved,
}: {
  paused?: boolean
  config?: typeof saved
}) {
  const ref = useRef<HTMLDivElement>(null)
  return (
    <div>
      <div ref={ref}>Form</div>
      <LoginBackground panelRef={ref} config={config} paused={paused} />
    </div>
  )
}

function animationEnvironment(reduced = false) {
  const frames = new Map<number, FrameRequestCallback>()
  let frameId = 0
  const request = vi.fn((callback: FrameRequestCallback) => {
    frames.set(++frameId, callback)
    return frameId
  })
  vi.stubGlobal("requestAnimationFrame", request)
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id))
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    setTransform: vi.fn(),
  } as unknown as CanvasRenderingContext2D)
  const remove = vi.fn()
  vi.stubGlobal("matchMedia", () => ({
    matches: reduced,
    addEventListener: vi.fn(),
    removeEventListener: remove,
  }))
  const tick = (time: number) =>
    act(() => {
      const callbacks = [...frames.values()]
      frames.clear()
      callbacks.forEach((callback) => {
        callback(time)
      })
    })
  return { frames, request, tick, remove }
}

describe("login background lifecycle", () => {
  it("animates, pauses without advancing time, resumes, and cleans up", () => {
    const env = animationEnvironment()
    const mounted = render(<Harness />)
    env.tick(1000)
    env.tick(1100)
    const beforePause = vi.mocked(drawBars).mock.lastCall?.[4]
    expect(beforePause).toBeGreaterThan(0)
    mounted.rerender(<Harness paused />)
    env.tick(1200)
    expect(env.frames.size).toBe(0)
    expect(vi.mocked(drawBars).mock.lastCall?.[4]).toBe(beforePause)
    mounted.rerender(<Harness />)
    env.tick(1300)
    env.tick(1400)
    expect(vi.mocked(drawBars).mock.lastCall?.[4]).toBeGreaterThan(
      beforePause ?? 0,
    )
    mounted.unmount()
    expect(env.frames.size).toBe(0)
    expect(env.remove).toHaveBeenCalledWith("change", expect.any(Function))
  })

  it("paints a static frame for reduced motion", () => {
    const env = animationEnvironment(true)
    render(<Harness />)
    env.tick(1000)
    expect(env.frames.size).toBe(0)
    expect(vi.mocked(drawBars).mock.lastCall?.[4]).toBe(0)
  })

  it("keeps the animation clock moving while controls are adjusted", () => {
    const env = animationEnvironment()
    const config = { ...saved, speed: 1 }
    const mounted = render(<Harness config={config} />)
    env.tick(1000)
    env.tick(1016)
    mounted.rerender(<Harness config={{ ...config, waveScale: 1.5 }} />)
    env.tick(1032)
    expect(vi.mocked(drawBars).mock.lastCall?.[4]).toBeCloseTo(0.032)
    expect(env.frames.size).toBe(1)
  })

  it("does not advance after the tab has been hidden", () => {
    const env = animationEnvironment()
    render(<Harness />)
    env.tick(1000)
    env.tick(1100)
    const before = vi.mocked(drawBars).mock.lastCall?.[4]
    const hidden = vi.spyOn(document, "hidden", "get").mockReturnValue(true)
    act(() => document.dispatchEvent(new Event("visibilitychange")))
    expect(env.frames.size).toBe(0)
    hidden.mockReturnValue(false)
    act(() => document.dispatchEvent(new Event("visibilitychange")))
    env.tick(9000)
    expect(vi.mocked(drawBars).mock.lastCall?.[4]).toBe(before)
  })
})
