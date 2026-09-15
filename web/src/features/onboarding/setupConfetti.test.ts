import { afterEach, beforeEach, expect, it, vi } from "vitest"
import {
  fireSetupConfetti,
  resetSetupConfetti,
} from "@/features/onboarding/setupConfetti"

const { confetti } = vi.hoisted(() => ({ confetti: vi.fn() }))
vi.mock("canvas-confetti", () => ({ default: confetti }))

beforeEach(() => {
  resetSetupConfetti()
  confetti.mockClear()
})
afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

function dialog() {
  const element = document.createElement("div")
  const backdrop = document.createElement("div")
  backdrop.style.zIndex = "100000"
  backdrop.append(element)
  document.body.append(backdrop)
  vi.spyOn(element, "getBoundingClientRect").mockReturnValue({
    left: 100,
    top: 120,
    width: 520,
    height: 240,
  } as DOMRect)
  return element
}

it("measures the dialog center after its entrance settles", async () => {
  const element = dialog()
  let settle!: () => void
  const finished = new Promise<Animation>((resolve) => {
    settle = () => resolve({} as Animation)
  })
  element.getAnimations = () => [{ finished } as Animation]
  const firing = fireSetupConfetti(element, new AbortController().signal)
  expect(confetti).not.toHaveBeenCalled()
  vi.mocked(element.getBoundingClientRect).mockReturnValue({
    left: 200,
    top: 140,
    width: 520,
    height: 240,
  } as DOMRect)
  settle()
  await firing
  expect(confetti).toHaveBeenCalledWith(
    expect.objectContaining({
      origin: { x: 460 / window.innerWidth, y: 260 / window.innerHeight },
      disableForReducedMotion: true,
      zIndex: 100001,
    }),
  )
})

it("does not celebrate a dismissed dialog or consume its next presentation", async () => {
  const element = dialog()
  const controller = new AbortController()
  const firing = fireSetupConfetti(element, controller.signal)
  controller.abort()
  await firing
  expect(confetti).not.toHaveBeenCalled()
  await fireSetupConfetti(element, new AbortController().signal)
  await fireSetupConfetti(element, new AbortController().signal)
  expect(confetti).toHaveBeenCalledTimes(1)
})

it("stays above canvas-confetti's own layer when no ancestor sets one", async () => {
  const element = document.createElement("div")
  document.body.append(element)
  await fireSetupConfetti(element, new AbortController().signal)
  expect(confetti).toHaveBeenCalledWith(
    expect.objectContaining({ zIndex: 101 }),
  )
})
