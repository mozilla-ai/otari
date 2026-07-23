import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "@/api/client";
import App from "@/App";
import { Provider } from "@/provider";

vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return { ...actual, apiFetch: vi.fn() };
});

vi.mock("@/pages/OverviewPage", async () => {
  await new Promise((resolve) => window.setTimeout(resolve, 20));
  return { OverviewIndex: () => <div>Lazy overview</div> };
});

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    window.location.hash = "";
  });

  it("shows a loading state while the current route loads", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1");
    vi.mocked(apiFetch).mockImplementation(async (path) => {
      if (path === "/dashboard-build.json") {
        return { build: "test-build" } as never;
      }
      if (path === "/v1/settings") {
        return { default_pricing: true, require_pricing: false } as never;
      }
      return [] as never;
    });

    render(
      <Provider>
        <App />
      </Provider>,
    );

    expect(screen.getByRole("status")).toHaveTextContent("Loading page…");
    expect(await screen.findByText("Lazy overview")).toBeInTheDocument();
  });
});
