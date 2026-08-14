import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppShell } from "@/components/AppShell";
import { Provider } from "@/provider";
import { renderWithRouter } from "@/test/router";

// jsdom has no layout engine, so `md:hidden` / responsive classes never take
// effect. The mobile-vs-desktop branch keys off window.matchMedia instead, which
// jsdom also does not implement, so tests drive it through this stub. The stub
// captures listeners so a viewport change can be simulated.
function mockMatchMedia(matches: boolean, options: { legacy?: boolean } = {}) {
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const mql: Record<string, unknown> = {
    matches,
    media: "",
    onchange: null,
    // Deprecated Safari < 14 API; always present.
    addListener: (cb: (event: MediaQueryListEvent) => void) => listeners.add(cb),
    removeListener: (cb: (event: MediaQueryListEvent) => void) => listeners.delete(cb),
    dispatchEvent: () => true,
  };
  // `legacy` omits the modern API so the component must fall back to addListener.
  if (!options.legacy) {
    mql.addEventListener = (_type: string, cb: (event: MediaQueryListEvent) => void) => listeners.add(cb);
    mql.removeEventListener = (_type: string, cb: (event: MediaQueryListEvent) => void) => listeners.delete(cb);
  }
  vi.stubGlobal("matchMedia", vi.fn().mockReturnValue(mql));
  return { mql, listeners };
}

// AppShell is the root route's component in the real tree, so it renders its
// page through an <Outlet>. Here it is the component under test and the pages
// are the router's children, which is what makes clicking a nav link swap them.
function renderShell() {
  return renderWithRouter(<div>OVERVIEW PAGE</div>, {
    url: "/",
    shell: (
      <Provider>
        <AppShell />
      </Provider>
    ),
    routes: [{ path: "/providers", element: <div>PROVIDERS PAGE</div> }],
  });
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
    const { container } = await renderShell();

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
    await renderShell();

    await user.click(screen.getByRole("button", { name: "Open navigation" }));
    await user.click(screen.getByRole("link", { name: "Providers" }));

    expect(await screen.findByText("PROVIDERS PAGE")).toBeInTheDocument();
    // Navigating closes the drawer so the page it landed on is not hidden behind it.
    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
  });

  it("marks only the current page's nav link as active", async () => {
    // Worth pinning because the router decides this, not the shell: a link is
    // active when its route is in the current match chain, so the index link is
    // lit only on the index. The react-router version needed an explicit `end`
    // to get that, and it would be easy to reintroduce something equivalent.
    mockMatchMedia(false);
    const user = userEvent.setup();
    await renderShell();

    const overview = screen.getByRole("link", { name: "Overview" });
    const providers = screen.getByRole("link", { name: "Providers" });
    expect(overview).toHaveAttribute("aria-current", "page");
    expect(providers).not.toHaveAttribute("aria-current");

    await user.click(providers);

    expect(await screen.findByText("PROVIDERS PAGE")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Providers" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Overview" })).not.toHaveAttribute("aria-current");
  });

  it("renders the resizable rail (not a drawer) on desktop", async () => {
    mockMatchMedia(false);
    const { container } = await renderShell();

    const aside = container.querySelector("aside");
    // Desktop keeps the in-flow, inline-width-driven rail rather than a fixed overlay.
    expect(aside?.className).not.toContain("fixed");
    expect(aside?.getAttribute("style")).toContain("width");
  });

  it("closes the drawer when the viewport grows past the mobile breakpoint", async () => {
    const { listeners } = mockMatchMedia(true);
    const user = userEvent.setup();
    await renderShell();

    await user.click(screen.getByRole("button", { name: "Open navigation" }));
    expect(screen.getByRole("button", { name: "Close navigation" })).toBeInTheDocument();

    // Simulate crossing to a desktop viewport.
    act(() => {
      listeners.forEach((cb) => cb({ matches: false } as MediaQueryListEvent));
    });

    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
  });

  it("closes the drawer on Escape and restores focus to the toggle", async () => {
    mockMatchMedia(true);
    const user = userEvent.setup();
    await renderShell();

    const toggle = screen.getByRole("button", { name: "Open navigation" });
    await user.click(toggle);
    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveAttribute("aria-expanded", "true");

    await user.keyboard("{Escape}");

    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
    // Focus returns to the trigger so a keyboard user is not dropped to the top.
    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveFocus();
  });

  it("closes the drawer when the backdrop is clicked", async () => {
    mockMatchMedia(true);
    const user = userEvent.setup();
    const { container } = await renderShell();

    await user.click(screen.getByRole("button", { name: "Open navigation" }));
    const backdrop = container.querySelector(".fixed.inset-0")!;
    expect(backdrop).toBeInTheDocument();

    await user.click(backdrop);

    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
  });

  it("marks the drawer inert while closed so its links leave the tab order", async () => {
    mockMatchMedia(true);
    const { container } = await renderShell();

    // Off-canvas and inert by default: the nav is not reachable until opened.
    expect(container.querySelector("aside")).toHaveAttribute("inert");
    expect(container.querySelector("aside")).toHaveAttribute("aria-label", "Navigation");
  });

  it("makes the background (header + main) inert while the drawer is open", async () => {
    mockMatchMedia(true);
    const user = userEvent.setup();
    const { container } = await renderShell();

    const header = container.querySelector("header")!;
    const main = container.querySelector("main")!;
    // Background is interactive until the modal drawer opens.
    expect(header).not.toHaveAttribute("inert");
    expect(main).not.toHaveAttribute("inert");

    await user.click(screen.getByRole("button", { name: "Open navigation" }));

    // aria-modal isn't universally honored, so inert is what actually keeps the
    // obscured page out of the tab order and the accessibility tree.
    expect(header).toHaveAttribute("inert");
    expect(main).toHaveAttribute("inert");
  });

  it("moves focus to the main region via the skip link without changing the route", async () => {
    mockMatchMedia(false);
    const user = userEvent.setup();
    const { container } = await renderShell();

    // The skip link is the first tab stop so keyboard users reach the page body
    // without traversing the whole nav.
    await user.tab();
    const skip = screen.getByRole("button", { name: "Skip to main content" });
    expect(skip).toHaveFocus();

    await user.keyboard("{Enter}");

    const main = container.querySelector("main")!;
    expect(main).toHaveFocus();
    // Focus moved without navigating away: the index route is still rendered.
    expect(screen.getByText("OVERVIEW PAGE")).toBeInTheDocument();
  });

  it("makes the skip link inert while the drawer is open, matching its target", async () => {
    mockMatchMedia(true);
    const user = userEvent.setup();
    await renderShell();

    const skip = screen.getByRole("button", { name: "Skip to main content" });
    // Live before the modal opens: it is the keyboard user's fast path to content.
    expect(skip).not.toHaveAttribute("inert");

    await user.click(screen.getByRole("button", { name: "Open navigation" }));

    // With the drawer open, main is inert, so focusing it would no-op. The skip
    // link goes inert too rather than sitting live in front of the modal as a
    // dead control an AT cursor reaches first.
    expect(skip).toHaveAttribute("inert");
  });

  it("hides the decorative nav icons from assistive tech", async () => {
    mockMatchMedia(false);
    const { container } = await renderShell();

    // Each nav link carries a visible text label, so its leading glyph is
    // decorative and must not be announced twice. Every SVG in the shell is marked
    // aria-hidden to match the rest of the codebase's convention.
    const icons = container.querySelectorAll("svg");
    expect(icons.length).toBeGreaterThan(0);
    icons.forEach((icon) => expect(icon).toHaveAttribute("aria-hidden"));
  });

  it("links to the bundled user guide from the sidebar footer", async () => {
    mockMatchMedia(false);
    await renderShell();

    // A footer link points operators at the guide bundled with this dashboard,
    // discoverable without hunting for a separate docs site.
    const guideLink = screen.getByRole("link", { name: "User guide" });
    expect(guideLink).toHaveAttribute("href", "/docs");
  });

  it("subscribes via the legacy matchMedia API when addEventListener is absent", async () => {
    // Safari < 14 exposes only addListener/removeListener; the shell must still
    // react to breakpoint changes rather than throwing on a missing method.
    const { listeners } = mockMatchMedia(true, { legacy: true });
    await renderShell();

    // The component registered through addListener, so the captured set is live.
    expect(listeners.size).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Open navigation" })).toBeInTheDocument();

    act(() => {
      listeners.forEach((cb) => cb({ matches: false } as MediaQueryListEvent));
    });

    // A desktop-width change still flips the layout off the mobile drawer.
    expect(screen.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
  });
});
