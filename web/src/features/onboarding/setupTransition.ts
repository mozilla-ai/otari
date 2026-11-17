import {
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react"

import { celebrationColors } from "@/features/onboarding/celebrationPalette"
import { usePrefersReducedMotion } from "@/shared/hooks/usePrefersReducedMotion"

import type { Palette, RGB } from "./vendor/distortion-waves/model"

/**
 * The landing: the first-run sheet carried into its success face by a
 * `distortion-waves` wave (vendored, see `vendor/distortion-waves/README.md`)
 * rising from the listening panel. The two faces share one `Dialog`, so the
 * setup face is copied as it stood, laid over the frame, and the wave carries
 * that copy into the success face rendered underneath.
 */

/** Marks the element the wave rises from: the listening panel. */
export const LANDING_SOURCE = "data-setup-listening"

/** Marks the still copy of the setup face while the wave carries it away. */
export const LANDING_COPY = "data-setup-landing-copy"

/** How long the success face may stay hidden while the wave gets ready. */
const READY_DEADLINE_MS = 1_500

/** Wider and taller than the phone sheet, whose height is the viewport's. */
const FLOATING_FRAME = "(width > 639px) and (height > 639px)"

const loadEngine = () =>
  Promise.all([
    import("./vendor/distortion-waves/controller"),
    import("./vendor/distortion-waves/model"),
  ])

/**
 * The wave takes channels, and a token can be written in any CSS color syntax,
 * so the browser parses it by painting one pixel. One it cannot parse leaves
 * the pixel clear and is dropped.
 */
function palette(singlePalette: (color: RGB) => Palette): Palette | undefined {
  const context = document
    .createElement("canvas")
    .getContext("2d", { willReadFrequently: true })
  if (!context) return undefined
  const colors = celebrationColors()
    .map((color): RGB | null => {
      context.clearRect(0, 0, 1, 1)
      context.fillStyle = "transparent"
      context.fillStyle = color
      context.fillRect(0, 0, 1, 1)
      const [red, green, blue, alpha] = context.getImageData(0, 0, 1, 1).data
      return alpha === 0 ? null : [red, green, blue]
    })
    .filter((color) => color !== null)
  if (colors.length === 0) return undefined
  return { ...singlePalette(colors[0]), colors }
}

/** The setup face as it stood the moment the request landed. */
interface Landing {
  /** The frame's height then, which the success face keeps. */
  height: number
  /** A still copy of the setup face, or none when the wave is not played. */
  copy: HTMLElement | null
  /** Scroll offsets inside the face, by element index, to put back on the copy. */
  scrolled: [number, number, number][]
}

/**
 * A still copy of the face, canvases included. The controller toggles `inert`
 * and `aria-hidden` on the layer it is given, so those go on the copy inside
 * it, which it never touches: nothing in the copy can be reached or read.
 */
function copyOf(face: HTMLElement): Pick<Landing, "copy" | "scrolled"> {
  const still = face.cloneNode(true) as HTMLElement
  const sources = face.querySelectorAll("canvas")
  still.querySelectorAll("canvas").forEach((canvas, index) => {
    const source = sources[index]
    if (source && source.width > 0 && source.height > 0) {
      canvas.getContext("2d")?.drawImage(source, 0, 0)
    }
  })
  for (const element of still.querySelectorAll("[id]")) {
    element.removeAttribute("id")
  }
  still.setAttribute("aria-hidden", "true")
  still.toggleAttribute("inert", true)

  const copy = document.createElement("div")
  copy.toggleAttribute(LANDING_COPY, true)
  copy.style.cssText =
    "position:absolute;inset:0;display:flex;flex-direction:column"
  copy.append(still)

  const scrolled: Landing["scrolled"] = []
  face.querySelectorAll("*").forEach((element, index) => {
    if (element.scrollTop || element.scrollLeft) {
      scrolled.push([index, element.scrollTop, element.scrollLeft])
    }
  })
  return { copy, scrolled }
}

function hide(face: HTMLElement) {
  face.style.opacity = "0"
  face.toggleAttribute("inert", true)
  face.setAttribute("aria-hidden", "true")
}

/** Gives the face its input back, leaving its opacity to whoever set it. */
function enable(face: HTMLElement) {
  face.style.pointerEvents = ""
  face.toggleAttribute("inert", false)
  face.removeAttribute("aria-hidden")
}

/**
 * Plays the wave from the copy of the setup face into `face`, and returns what
 * tears it down. The success face takes input as soon as the front has crossed
 * it, and is revealed whatever happens to the wave: when it ends, when it is
 * not ready in time, when the GPU refuses it, and when the frame goes mid-run.
 */
function carry(
  frame: HTMLElement,
  face: HTMLElement,
  copy: HTMLElement,
  scrolled: Landing["scrolled"],
): () => void {
  copy.style.backgroundColor = getComputedStyle(frame).backgroundColor
  frame.append(copy)
  const elements = copy.querySelectorAll("*")
  for (const [index, top, left] of scrolled) {
    const element = elements[index + 1]
    if (element) {
      element.scrollTop = top
      element.scrollLeft = left
    }
  }
  hide(face)

  let wave: { destroy(): void } | null = null
  let watching = 0
  let isDone = false
  const finish = () => {
    if (isDone) return
    isDone = true
    window.clearTimeout(deadline)
    cancelAnimationFrame(watching)
    wave?.destroy()
    copy.remove()
    face.style.opacity = ""
    enable(face)
  }
  const deadline = window.setTimeout(finish, READY_DEADLINE_MS)

  const play = async () => {
    const [{ WaveController }, { defaultStyle, singlePalette }] =
      await loadEngine()
    if (isDone) return
    const controller = new WaveController(frame, copy, face)
    wave = controller
    const options = {
      type: "vertical" as const,
      palette: palette(singlePalette),
      // Pictured once, below: the copy never changes.
      prewarm: false,
      live: false,
      onFinish: finish,
    }
    controller.update({ ...options, revealed: false })
    // Clicks pass through the copy to the success face once it takes input.
    copy.style.pointerEvents = "none"
    await controller.prepare()
    if (isDone) return
    window.clearTimeout(deadline)
    controller.update({
      ...options,
      revealed: true,
      source:
        copy.querySelector(`[${LANDING_SOURCE}]`) ??
        copy.querySelector("footer"),
    })
    // The controller hands the face over only when the color has left its
    // edges, which on this type is the end of the run. The front has crossed
    // the frame well before, so input goes to the face then. Read off the
    // wave's own frames, so a paused tab keeps the two in step.
    const style = defaultStyle("vertical")
    const crossed = style.anticipation + style.travel * 0.6
    const watch = () => {
      const { progress, duration } = controller.debug
      if (progress * duration >= crossed) enable(face)
      else watching = requestAnimationFrame(watch)
    }
    watching = requestAnimationFrame(watch)
  }
  // The library throws when the GPU will not compile its shaders.
  play().catch(finish)

  return finish
}

/**
 * Settles the frame on its success face, plays the wave when there is a copy
 * to carry, and returns what undoes both.
 */
function land(
  frame: HTMLElement,
  face: HTMLElement,
  { height, copy, scrolled }: Landing,
): () => void {
  // The body is the same node in both faces, so it would keep the sheet's
  // scroll, and the header its rule.
  for (const element of face.querySelectorAll("*")) {
    if (element.scrollTop) element.scrollTop = 0
  }
  // Held at the sheet's height so the frame does not resize, for as long as it
  // floats: `max-height` still caps it, and the phone sheet takes the
  // viewport's height, even when the window becomes one after the landing.
  const floating = window.matchMedia(FLOATING_FRAME)
  const hold = () => {
    frame.style.height = floating.matches ? `${height}px` : ""
  }
  hold()
  floating.addEventListener("change", hold)
  // The frame did not reopen, so nothing announces the new title; focus does.
  frame.focus({ preventScroll: true })
  const stop = copy ? carry(frame, face, copy, scrolled) : undefined
  return () => {
    floating.removeEventListener("change", hold)
    stop?.()
  }
}

/**
 * Whether the success face should render, and the wave that carries the setup
 * face into it once it does.
 *
 * The answer lags `landed` by one commit, taken before the browser paints: the
 * setup face is still on screen in that commit, the only moment it can be
 * copied. Under reduced motion the faces swap at once and the engine is never
 * loaded.
 */
export function useLandingTransition(
  faceRef: RefObject<HTMLDivElement | null>,
  landed: boolean,
): boolean {
  const prefersReducedMotion = usePrefersReducedMotion()
  const [showsSuccess, setShowsSuccess] = useState(landed)
  const landing = useRef<Landing | null>(null)

  // Fetched while the sheet waits. A failure here is tried again at the
  // landing, which shows the payoff without the wave if it fails there too.
  useEffect(() => {
    if (!landed && !prefersReducedMotion) loadEngine().catch(() => undefined)
  }, [landed, prefersReducedMotion])

  useLayoutEffect(() => {
    if (!landed || showsSuccess) return
    const face = faceRef.current
    const frame = face?.parentElement
    if (face && frame) {
      landing.current = {
        height: frame.getBoundingClientRect().height,
        ...(prefersReducedMotion ? { copy: null, scrolled: [] } : copyOf(face)),
      }
    }
    setShowsSuccess(true)
  }, [faceRef, landed, showsSuccess, prefersReducedMotion])

  useLayoutEffect(() => {
    const next = landing.current
    const face = faceRef.current
    const frame = face?.parentElement
    if (!showsSuccess || !next || !face || !frame) return
    landing.current = null
    return land(frame, face, next)
  }, [faceRef, showsSuccess])

  return showsSuccess
}
