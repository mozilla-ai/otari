import { render, screen, waitFor } from "@testing-library/react"
import { useRef } from "react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import {
  LANDING_COPY,
  LANDING_SOURCE,
  useLandingTransition,
} from "@/features/onboarding/setupTransition"

type Options = {
  revealed?: boolean
  live?: boolean
  source?: Element | null
  type?: string
  onFinish?: (revealed: boolean) => void
}

type FakeWave = {
  root: HTMLElement
  content: HTMLElement
  revealed: HTMLElement
  updates: Options[]
  prepare: ReturnType<typeof vi.fn>
  destroy: ReturnType<typeof vi.fn>
  debug: { progress: number; duration: number }
}

const { waves, WaveController, prepared } = vi.hoisted(() => {
  const waves: FakeWave[] = []
  const prepared = { next: () => Promise.resolve() }
  const WaveController = vi.fn(function (
    this: unknown,
    root: HTMLElement,
    content: HTMLElement,
    revealed: HTMLElement,
  ) {
    const wave = {
      root,
      content,
      revealed,
      updates: [] as Options[],
      update(options: Options) {
        wave.updates.push(options)
      },
      prepare: vi.fn(() => prepared.next()),
      destroy: vi.fn(),
      debug: { progress: 0, duration: 3.62 },
    }
    waves.push(wave)
    return wave
  })
  return { waves, WaveController, prepared }
})
vi.mock("@/features/onboarding/vendor/distortion-waves/controller", () => ({
  WaveController,
}))

/** The sheet and its payoff as two faces of one frame, like `SetupSheet`. */
function Sheet({ landed }: { landed: boolean }) {
  const faceRef = useRef<HTMLDivElement>(null)
  const showsSuccess = useLandingTransition(faceRef, landed)
  return (
    <Dialog
      isOpen
      onOpenChange={vi.fn()}
      faceRef={faceRef}
      title={showsSuccess ? "It landed" : "Waiting"}
      status={
        showsSuccess ? (
          <p>Setup complete</p>
        ) : (
          <div {...{ [LANDING_SOURCE]: "" }}>Listening</div>
        )
      }
    >
      <DialogSection>
        <p>{showsSuccess ? "the payoff" : "the example"}</p>
      </DialogSection>
    </Dialog>
  )
}

/** Reduced motion or not, on a floating frame or the phone sheet. */
function media({ reduce = false, phone = false } = {}) {
  const size = { phone, listeners: new Set<() => void>() }
  vi.spyOn(window, "matchMedia").mockImplementation((query: string) => {
    const isSize = query.includes("639px")
    return {
      get matches() {
        return isSize ? !size.phone : reduce
      },
      media: query,
      addEventListener: (_: string, listener: () => void) => {
        if (isSize) size.listeners.add(listener)
      },
      removeEventListener: (_: string, listener: () => void) => {
        size.listeners.delete(listener)
      },
    } as unknown as MediaQueryList
  })
  /** Resizes the window across the phone sheet's threshold. */
  return (toPhone: boolean) => {
    size.phone = toPhone
    for (const listener of size.listeners) listener()
  }
}

/** Renders the sheet waiting, then lands the request, and returns the frame. */
async function land() {
  const view = render(<Sheet landed={false} />)
  const frame = await screen.findByRole("dialog", { name: "Waiting" })
  view.rerender(<Sheet landed />)
  return { ...view, frame }
}

function copyIn(frame: HTMLElement) {
  return frame.querySelector<HTMLElement>(`[${LANDING_COPY}]`)
}

function faceIn(frame: HTMLElement) {
  return frame.querySelector(`:scope > :not([${LANDING_COPY}]) header`)!
    .parentElement as HTMLElement
}

beforeEach(() => {
  waves.length = 0
  WaveController.mockClear()
  prepared.next = () => Promise.resolve()
  media()
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it("lays a still copy of the sheet over the frame, with the payoff hidden under it", async () => {
  prepared.next = () => new Promise(() => undefined)
  const { frame } = await land()

  const copy = copyIn(frame)
  expect(copy).toHaveTextContent("the example")
  expect(faceIn(frame)).toHaveTextContent("the payoff")
  expect(faceIn(frame)).toHaveAttribute("aria-hidden", "true")
  expect(faceIn(frame)).toHaveAttribute("inert")
})

it("keeps the copy unreachable while the wave treats it as the live view", async () => {
  const { frame } = await land()
  await waitFor(() => expect(waves).toHaveLength(1))

  // What the controller does to the layer it is showing.
  const copy = copyIn(frame)!
  copy.toggleAttribute("inert", false)
  copy.removeAttribute("aria-hidden")

  expect(copy.firstElementChild).toHaveAttribute("inert")
  expect(copy.firstElementChild).toHaveAttribute("aria-hidden", "true")
  expect(copy.style.pointerEvents).toBe("none")
})

it("hands the payoff its input once the front has crossed it, before the color fades", async () => {
  const { frame } = await land()
  await waitFor(() => expect(waves[0]?.updates.at(-1)?.revealed).toBe(true))
  expect(faceIn(frame)).toHaveAttribute("inert")

  // Past the anticipation and the front's travel, with the color still on
  // the edges: the run has not finished.
  waves[0].debug.progress = 0.6

  await waitFor(() => expect(faceIn(frame)).not.toHaveAttribute("inert"))
  expect(copyIn(frame)).not.toBeNull()
})

it("carries the copy into the payoff, rising from the listening panel", async () => {
  const { frame } = await land()

  await waitFor(() => expect(waves).toHaveLength(1))
  const [wave] = waves
  const copy = copyIn(frame)!
  expect(wave.root).toBe(frame)
  expect(wave.content).toBe(copy)
  expect(wave.revealed).toBe(faceIn(frame))
  await waitFor(() =>
    expect(wave.updates.at(-1)).toMatchObject({
      type: "vertical",
      revealed: true,
      source: copy.querySelector(`[${LANDING_SOURCE}]`),
    }),
  )
  // Both views pictured before the wave sets out, so the lens is there from
  // its first frame.
  expect(wave.updates[0]).toMatchObject({ revealed: false, live: false })
  expect(wave.prepare).toHaveBeenCalledOnce()
})

it("takes the copy away and leaves the payoff live once the wave is done", async () => {
  const { frame } = await land()
  await waitFor(() => expect(waves[0]?.updates.at(-1)?.revealed).toBe(true))

  waves[0].updates[0].onFinish?.(true)

  expect(copyIn(frame)).toBeNull()
  expect(faceIn(frame)).not.toHaveAttribute("aria-hidden")
  expect(faceIn(frame)).not.toHaveAttribute("inert")
  expect(waves[0].destroy).toHaveBeenCalled()
})

it("shows the payoff without a wave when the GPU refuses it", async () => {
  prepared.next = () => Promise.reject(new Error("vertex shader: undefined"))
  const { frame } = await land()

  await waitFor(() => expect(copyIn(frame)).toBeNull())
  expect(faceIn(frame)).not.toHaveAttribute("inert")
  expect(screen.getByRole("dialog", { name: "It landed" })).toBeInTheDocument()
})

it("does not hold the payoff back for a wave that is slow to get ready", async () => {
  prepared.next = () => new Promise(() => undefined)
  const view = render(<Sheet landed={false} />)
  const frame = await screen.findByRole("dialog", { name: "Waiting" })
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
  view.rerender(<Sheet landed />)
  expect(copyIn(frame)).not.toBeNull()

  await vi.advanceTimersByTimeAsync(1_500)

  expect(copyIn(frame)).toBeNull()
  expect(faceIn(frame)).not.toHaveAttribute("inert")
})

it("tears the wave down when the frame goes mid-run", async () => {
  const { unmount } = await land()
  await waitFor(() => expect(waves).toHaveLength(1))

  unmount()

  expect(waves[0].destroy).toHaveBeenCalled()
})

it("keeps the frame at the sheet's height, so the payoff does not resize it", async () => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    function (this: HTMLElement) {
      return {
        height: this.getAttribute("role") === "dialog" ? 660 : 0,
      } as DOMRect
    },
  )
  const { frame } = await land()

  expect(frame.style.height).toBe("660px")
})

it("lets go of the held height when the window becomes the phone sheet", async () => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    function (this: HTMLElement) {
      return {
        height: this.getAttribute("role") === "dialog" ? 660 : 0,
      } as DOMRect
    },
  )
  const resize = media()
  const { frame } = await land()
  expect(frame.style.height).toBe("660px")

  resize(true)
  expect(frame.style.height).toBe("")

  resize(false)
  expect(frame.style.height).toBe("660px")
})

it("leaves the phone sheet's height to the viewport", async () => {
  media({ phone: true })
  const { frame } = await land()

  expect(frame.style.height).toBe("")
})

it("opens the payoff at its top when the sheet had been scrolled", async () => {
  const view = render(<Sheet landed={false} />)
  const frame = await screen.findByRole("dialog", { name: "Waiting" })
  const body = faceIn(frame).querySelector(".overflow-y-auto") as HTMLElement
  body.scrollTop = 120
  expect(body.scrollTop).toBe(120)

  view.rerender(<Sheet landed />)

  expect(body.scrollTop).toBe(0)
})

it("moves focus to the frame, so the new title is announced", async () => {
  const { frame } = await land()

  expect(frame).toHaveFocus()
})

it("swaps the faces at once under reduced motion, without loading the wave", async () => {
  media({ reduce: true })
  const { frame } = await land()

  expect(copyIn(frame)).toBeNull()
  expect(faceIn(frame)).toHaveTextContent("the payoff")
  expect(faceIn(frame)).not.toHaveAttribute("inert")
  expect(WaveController).not.toHaveBeenCalled()
})

it("shows the payoff straight away when it was already the state on mount", async () => {
  render(<Sheet landed />)

  expect(
    await screen.findByRole("dialog", { name: "It landed" }),
  ).toBeInTheDocument()
  expect(WaveController).not.toHaveBeenCalled()
})
