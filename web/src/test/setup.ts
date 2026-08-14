import "@testing-library/jest-dom/vitest";

// recharts' ResponsiveContainer measures its parent through ResizeObserver, which
// jsdom does not implement. Provide a minimal stub that reports a fixed size on
// observe so charts mount (and render their SVG) in component tests instead of
// throwing on `new ResizeObserver(...)`.
class ResizeObserverStub {
  constructor(private readonly callback: ResizeObserverCallback) {}
  observe(target: Element): void {
    const rect = { width: 800, height: 300, top: 0, left: 0, right: 800, bottom: 300, x: 0, y: 0, toJSON() {} };
    this.callback([{ target, contentRect: rect } as ResizeObserverEntry], this as unknown as ResizeObserver);
  }
  unobserve(): void {}
  disconnect(): void {}
}

// Only install the stub when the environment lacks a ResizeObserver, so a future
// jsdom (or another polyfill) that provides one is not clobbered.
if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}

// The router scrolls the window to the top of each new location, which jsdom
// does not implement and reports as an unhandled "Not implemented" on every
// navigation. Nothing in the dashboard scrolls the window (the page body is a
// `main` with its own overflow), so a no-op is a faithful stand-in and keeps the
// real failures visible in test output.
if (typeof window.scrollTo !== "function" || !("__stubbed" in window.scrollTo)) {
  window.scrollTo = Object.assign(() => undefined, { __stubbed: true });
}

// jsdom has no canvas 2D context, no toBlob, no object URLs, no ClipboardItem and
// no document.fonts, so nothing in lib/shareImage.ts can run for real here. These
// stubs let the share panel mount and its wiring be asserted; the claim that the
// PNG itself is correct is only provable in Playwright (see web/e2e).
if (typeof URL.createObjectURL !== "function") {
  URL.createObjectURL = () => "blob:stub";
}
if (typeof URL.revokeObjectURL !== "function") {
  URL.revokeObjectURL = () => undefined;
}
