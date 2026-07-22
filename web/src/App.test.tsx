import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "@/App";
import { Provider } from "@/provider";

vi.mock("@/pages/OverviewPage", async () => {
  await new Promise((resolve) => window.setTimeout(resolve, 20));
  return { OverviewIndex: () => <div>Lazy overview</div> };
});

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
    window.location.hash = "";
  });

  it("shows a loading state while the current route loads", async () => {
    window.sessionStorage.setItem("otari.dashboard.masterKey", "test-master-key");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify([])));

    render(
      <Provider>
        <App />
      </Provider>,
    );

    expect(screen.getByRole("status")).toHaveTextContent("Loading page…");
    expect(await screen.findByText("Lazy overview")).toBeInTheDocument();
  });
});
