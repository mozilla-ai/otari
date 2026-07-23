import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/AppShell";
import { Provider } from "@/provider";

// jsdom has no layout engine, so `md:hidden` / responsive classes never take
// effect. The mobile-vs-desktop branch keys off window.matchMedia instead, which
// jsdom also does not implement, so tests drive it through this stub. The stub
// captures listeners so a viewport change can be simulated.
function mockMatchMedia(matches: boolean) {
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const mql = {
    matches,
    media: "",
    onchange: null,
    addEventListener: (_type: string, cb: (event: MediaQueryListEvent) => void) => listeners.add(cb),
    removeEventListener: (_type: string, cb: (event: MediaQueryListEvent) => void) => listeners.delete(cb),
    addListener: (cb: (event: MediaQueryListEvent) => void) => listeners.add(cb),
    removeListener: (cb: (event: MediaQueryListEvent) => void) => listeners.delete(cb),
    dispatchEvent: () => true,
  };
  vi.stubGlobal("matchMedia", vi.fn().mockReturnValue(mql));
  return { mql, listeners };
}

function renderShell() {
  return render(
    <Provider>
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<div>OVERVIEW PAGE</div>} />
            <Route path="providers" element={<div>PROVIDERS PAGE</div>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </Provider>,
  );
}

describe("AppShell responsive layout", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("keeps the sidebar an off-canvas drawer on mobile, toggled from the header", async () => {
    mockMatchMedia(true);
    const user = userEvent.setup();
    const { container } = renderShell();

    const aside = container.querySelector("aside");
    // Off-canvas by default so it does not squash the page's content.
    expect(aside?.className).toContain("fixed");
    expect(aside?.className).toContain("-translate-x-full");

    const toggle = screen.getByRole("button", { name: "Open navigation" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    await user.click(toggle);

    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveAttribute("aria-expanded", "true");
    expect(aside?.className).toContain("translate-x-0");
  });

  it("dismisses the mobile drawer after navigating to a destination", async () => {
    mockMatchMedia(true);
    const user = userEvent.setup();
    renderShell();

    await user.click(screen.getByRole("button", { name: "Open navigation" }));
    await user.click(screen.getByRole("link", { name: "Providers" }));

    expect(await screen.findByText("PROVIDERS PAGE")).toBeInTheDocument();
    // Navigating closes the drawer so the page it landed on is not hidden behind it.
    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
  });

  it("renders the resizable rail (not a drawer) on desktop", () => {
    mockMatchMedia(false);
    const { container } = renderShell();

    const aside = container.querySelector("aside");
    // Desktop keeps the in-flow, inline-width-driven rail rather than a fixed overlay.
    expect(aside?.className).not.toContain("fixed");
    expect(aside?.getAttribute("style")).toContain("width");
  });

  it("closes the drawer when the viewport grows past the mobile breakpoint", async () => {
    const { listeners } = mockMatchMedia(true);
    const user = userEvent.setup();
    renderShell();

    await user.click(screen.getByRole("button", { name: "Open navigation" }));
    expect(screen.getByRole("button", { name: "Close navigation" })).toBeInTheDocument();

    // Simulate crossing to a desktop viewport.
    act(() => {
      listeners.forEach((cb) => cb({ matches: false } as MediaQueryListEvent));
    });

    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
  });
});
