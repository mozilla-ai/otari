import "@testing-library/jest-dom/vitest"

// recharts' ResponsiveContainer measures its parent through ResizeObserver, which
// jsdom does not implement. Provide a minimal stub that reports a fixed size on
// observe so charts mount (and render their SVG) in component tests instead of
// throwing on `new ResizeObserver(...)`.
class ResizeObserverStub {
  constructor(private readonly callback: ResizeObserverCallback) {}
  observe(target: Element): void {
    const rect = {
      width: 800,
      height: 300,
      top: 0,
      left: 0,
      right: 800,
      bottom: 300,
      x: 0,
      y: 0,
      toJSON() {},
    }
    this.callback(
      [{ target, contentRect: rect } as ResizeObserverEntry],
      this as unknown as ResizeObserver,
    )
  }
  unobserve(): void {}
  disconnect(): void {}
}

// Only install the stub when the environment lacks a ResizeObserver, so a future
// jsdom (or another polyfill) that provides one is not clobbered.
if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver =
    ResizeObserverStub as unknown as typeof ResizeObserver
}

// The router scrolls the window to the top of each new location, which jsdom
// does not implement and reports as an unhandled "Not implemented" on every
// navigation. Nothing in the dashboard scrolls the window (the page body is a
// `main` with its own overflow), so a no-op is a faithful stand-in and keeps the
// real failures visible in test output.
if (
  typeof window.scrollTo !== "function" ||
  !("__stubbed" in window.scrollTo)
) {
  window.scrollTo = Object.assign(() => undefined, { __stubbed: true })
}

// jsdom implements no scrolling at all, so `scrollIntoView` is absent on every
// element and a component that scrolls its own content into view throws in a
// passive effect. A no-op is faithful (there is no viewport to scroll) and it is
// spy-able, which is how the call itself is asserted.
if (typeof Element.prototype.scrollIntoView !== "function") {
  Element.prototype.scrollIntoView = () => undefined
}

// jsdom has no canvas 2D context, no toBlob, no object URLs, no ClipboardItem and
// no document.fonts, so nothing in lib/shareImage.ts can run for real here. These
// stubs let the share panel mount and its wiring be asserted; the claim that the
// PNG itself is correct is only provable in Playwright (see web/e2e).
if (typeof URL.createObjectURL !== "function") {
  URL.createObjectURL = () => "blob:stub"
}
if (typeof URL.revokeObjectURL !== "function") {
  URL.revokeObjectURL = () => undefined
}

// jsdom implements no `matchMedia`, which is why `useTheme`, `AppShell` and
// `usePrefersReducedMotion` all guard for its absence. A component that does not
// guard crashes here and nowhere else, and the crash is easy to introduce
// without noticing: `LoginBackground` reaches its `matchMedia` call only after
// `getContext("2d")` returns something, so while that answered null this file's
// own canvas stub was all that stood between the suite and this error.
//
// Answering "nothing matches" is the honest default. A dark theme and a reduced
// motion preference are both opt-in, so every query this product asks is false
// unless a test says otherwise, and a test that wants otherwise stubs the global
// itself.
if (typeof window.matchMedia !== "function") {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  })) as typeof window.matchMedia
}

// jsdom's `getContext` returns null, which is a faithful "no canvas here" and is
// also not what a canvas library expects: `canvas-confetti` calls `clearRect` on
// the result inside a `requestAnimationFrame`, so the null lands as an unhandled
// exception in a later tick rather than as a failed assertion in the test that
// caused it. The setup guide draws two canvases (the orb and the success
// burst), and neither is provable here: what a component test can show is that
// they mount and that the screen around them is right.
//
// A proxy rather than a hand-written context: the two libraries between them
// reach for a few dozen methods and properties, and a list of them would be a
// list to keep updating. Every method is a no-op, every property reads as
// undefined and accepts a write, and the handful that must return an object
// (the gradient and pattern factories, `measureText`) return one shaped enough
// to be chained off.
function stubCanvasContext(): unknown {
  const noop = () => undefined
  const target: Record<string, unknown> = {
    canvas: undefined,
    measureText: () => ({ width: 0 }),
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop }),
    createPattern: () => null,
    getImageData: () => ({ data: new Uint8ClampedArray(4) }),
  }
  return new Proxy(target, {
    get(store, property: string) {
      if (property in store) return store[property]
      // Unseen names are methods until something writes to one: a bare
      // property read (`ctx.globalAlpha`) then reads back what was written.
      return noop
    },
    set(store, property: string, value) {
      store[property] = value
      return true
    },
  })
}

// Decided once, not per call. jsdom writes a "Not implemented" line to the
// virtual console every time its own `getContext` is reached, so a wrapper that
// tried the real one first would print that line for every frame the two
// canvases draw. Probing once leaves a single line and installs the stub over
// the top; an environment that does support canvas keeps its own.
if (typeof HTMLCanvasElement !== "undefined") {
  const probe = document.createElement("canvas")
  if (probe.getContext("2d") === null) {
    HTMLCanvasElement.prototype.getContext = function getContext(
      this: HTMLCanvasElement,
    ) {
      const stub = stubCanvasContext() as { canvas: HTMLCanvasElement }
      stub.canvas = this
      return stub as unknown as RenderingContext
    } as HTMLCanvasElement["getContext"]
  }
}
