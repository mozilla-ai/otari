import {
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react"

import { usePrefersReducedMotion } from "@/shared/hooks/usePrefersReducedMotion"

import type { Palette, RGB } from "./vendor/distortion-waves/model"

/**
 * The landing: the first-run sheet carried into its success face by a wave
 * that rises from the listening panel, the moment the first request is seen.
 *
 * `distortion-waves` (vendored, see `vendor/distortion-waves/README.md`) bends
 * a picture of one view into another inside the same element. The sheet and
 * its success are two faces of one `Dialog`, so the frame never closes or
 * resizes: the setup face is copied as it stood, laid over the frame, and the
 * wave carries that copy into the success face rendered underneath.
 */

/** Marks the element the wave rises from: the listening panel. */
export const LANDING_SOURCE = "data-setup-listening"

/** Marks the still copy of the setup face while the wave carries it away. */
export const LANDING_COPY = "data-setup-landing-copy"

/**
 * How long the success face may stay hidden while the wave gets ready. The
 * picture of a view usually takes a frame or two, but it can wait on a font,
 * and the payoff is not worth delaying for an effect.
 */
const READY_DEADLINE_MS = 1_500

const loadEngine = () =>
  Promise.all([
    import("./vendor/distortion-waves/controller"),
    import("./vendor/distortion-waves/model"),
  ])

/**
 * The palette, read off the running theme rather than written here.
 *
 * A hex at a call site cannot follow a theme and there are two of them, which
 * is the design system's layer rule and is enforced over this file. The two
 * brand steps and the one green: the library's own palette is seven hues, and
 * a celebration in seven hues is not this product.
 */
const TOKENS = [
  "--color-primary",
  "--color-primary-hover",
  "--color-success",
  "--color-chart-ramp-4",
]

/**
 * The wave takes channels, and a token can be written in any CSS color syntax.
 * Painting one pixel lets the browser do the parsing, so a token that moves to
 * `oklch()` keeps working. A value it cannot parse leaves the pixel clear and
 * is dropped.
 */
function rgbOf(color: string, context: CanvasRenderingContext2D): RGB | null {
  context.clearRect(0, 0, 1, 1)
  context.fillStyle = "transparent"
  context.fillStyle = color
  context.fillRect(0, 0, 1, 1)
  const [red, green, blue, alpha] = context.getImageData(0, 0, 1, 1).data
  return alpha === 0 ? null : [red, green, blue]
}

function paletteFromTheme(
  singlePalette: (color: RGB) => Palette,
): Palette | undefined {
  const context = document
    .createElement("canvas")
    .getContext("2d", { willReadFrequently: true })
  if (!context) return undefined
  const styles = getComputedStyle(document.documentElement)
  const colors = TOKENS.map((token) =>
    rgbOf(styles.getPropertyValue(token).trim(), context),
  ).filter((color) => color !== null)
  if (colors.length === 0) return undefined
  // The fringes either side of the front are the primary, lighter ahead and
  // deeper behind, the way the library derives them for a single color.
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
 * A still copy of the face: the DOM as it renders now, canvases included,
 * with nothing in it that could be reached, focused or read twice.
 */
function copyOf(face: HTMLElement): Pick<Landing, "copy" | "scrolled"> {
  const copy = face.cloneNode(true) as HTMLElement
  const sources = face.querySelectorAll("canvas")
  copy.querySelectorAll("canvas").forEach((canvas, index) => {
    const source = sources[index]
    if (source && source.width > 0 && source.height > 0) {
      canvas.getContext("2d")?.drawImage(source, 0, 0)
    }
  })
  for (const element of copy.querySelectorAll("[id]")) {
    element.removeAttribute("id")
  }
  copy.setAttribute("aria-hidden", "true")
  copy.toggleAttribute("inert", true)
  copy.toggleAttribute(LANDING_COPY, true)
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

function reveal(face: HTMLElement) {
  face.style.opacity = ""
  face.style.pointerEvents = ""
  face.toggleAttribute("inert", false)
  face.removeAttribute("aria-hidden")
}

/**
 * Plays the wave from the copy of the setup face into `face`, and returns what
 * tears it down. Whatever happens to the wave, the success face is revealed:
 * when it finishes, when it cannot start in time, when the GPU refuses it, and
 * when the frame goes away mid-run.
 */
function carry(
  frame: HTMLElement,
  face: HTMLElement,
  copy: HTMLElement,
  scrolled: Landing["scrolled"],
): () => void {
  // Over the frame, where the sheet was, on the frame's own fill: the copy
  // stands for the sheet until the wave hands over.
  copy.style.position = "absolute"
  copy.style.inset = "0"
  copy.style.backgroundColor = getComputedStyle(frame).backgroundColor
  frame.append(copy)
  const elements = copy.querySelectorAll("*")
  for (const [index, top, left] of scrolled) {
    elements[index]?.scrollTo(left, top)
  }
  hide(face)

  let wave: { destroy(): void } | null = null
  let isDone = false
  const finish = () => {
    if (isDone) return
    isDone = true
    window.clearTimeout(deadline)
    wave?.destroy()
    copy.remove()
    reveal(face)
  }
  const deadline = window.setTimeout(finish, READY_DEADLINE_MS)

  const play = async () => {
    const [{ WaveController }, { singlePalette }] = await loadEngine()
    if (isDone) return
    const controller = new WaveController(frame, copy, face)
    wave = controller
    const options = {
      type: "vertical" as const,
      palette: paletteFromTheme(singlePalette),
      // Pictured once, below, and not again: the copy never changes, and the
      // success face is shown as it was when the wave set out.
      prewarm: false,
      live: false,
      onFinish: finish,
    }
    controller.update({ ...options, revealed: false })
    // Both pictures first, so the lens is there from the wave's first frame.
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
  }
  // The library throws when the GPU will not compile its shaders (it falls
  // back to a crossfade only when WebGL 2 is missing outright). The success
  // face is what matters, so a wave that cannot run is not shown.
  play().catch(finish)

  return finish
}

/**
 * Whether the success face should render, and the wave that carries the
 * setup face into it once it does.
 *
 * `landed` turns true when the first request is seen. The answer lags it by
 * one commit, taken before the browser paints: the setup face is still on
 * screen in that commit, which is the only moment it can be copied.
 *
 * Under reduced motion the faces swap at once, and the engine is never loaded.
 */
export function useLandingTransition(
  faceRef: RefObject<HTMLDivElement | null>,
  landed: boolean,
): boolean {
  const prefersReducedMotion = usePrefersReducedMotion()
  const [showsSuccess, setShowsSuccess] = useState(landed)
  const landing = useRef<Landing | null>(null)

  // Fetched while the sheet waits, so the landing never waits on the network.
  useEffect(() => {
    if (!landed && !prefersReducedMotion) void loadEngine()
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
    // The success face keeps the sheet's height, so the frame does not
    // resize: it is designed to fill the sheet rather than shrink to itself.
    // Capped by the same viewport budget the frame's own maximum uses.
    frame.style.minHeight = `min(${next.height}px, calc(var(--visual-viewport-height, 100dvh) - 15rem))`
    // The frame did not reopen, so nothing announces the new title. Focusing
    // the frame does, and takes focus off a control that just unmounted.
    frame.focus({ preventScroll: true })
    if (!next.copy) return
    return carry(frame, face, next.copy, next.scrolled)
  }, [faceRef, showsSuccess])

  return showsSuccess
}
