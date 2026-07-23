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
