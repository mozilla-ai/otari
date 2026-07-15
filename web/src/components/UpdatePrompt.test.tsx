import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import { UpdatePrompt } from "@/components/UpdatePrompt";

// The build the fake gateway is currently serving. Tests flip it to stand in
// for a deploy landing under an already-open tab.
let servedBuild = "build-a";

function mockGateway() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
    return new Response(JSON.stringify({ build: servedBuild, version: "1.0.0" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
  return client;
}

// Resolves once the tab has actually observed a build, which is what fixes the
// version it considers itself to be running. Waiting on fetch being *called* is
// not enough: the response may not have landed yet, and the tab would then
// adopt whatever build arrives first, including a newer one.
async function waitForLoadedBuild(client: QueryClient, build: string) {
  await waitFor(() => expect(client.getQueryData(["build"])).toMatchObject({ build }));
}

describe("UpdatePrompt", () => {
  beforeEach(() => {
    servedBuild = "build-a";
    setMasterKey("test-master-key");
  });
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("stays quiet while the served build matches the loaded one", async () => {
    mockGateway();

    const client = renderWithClient(<UpdatePrompt />);
    await waitForLoadedBuild(client, "build-a");
    await act(() => client.refetchQueries({ queryKey: ["build"] }));

    expect(screen.queryByText(/An update is available/)).not.toBeInTheDocument();
  });

  it("offers a reload once the gateway serves a different build", async () => {
    const reload = vi.fn();
    // jsdom's location.reload cannot be spied on directly.
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload },
    });
    mockGateway();

    const client = renderWithClient(<UpdatePrompt />);
    await waitForLoadedBuild(client, "build-a");

    servedBuild = "build-b";
    await act(() => client.refetchQueries({ queryKey: ["build"] }));

    expect(await screen.findByText(/An update is available/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Update now" }));
    expect(reload).toHaveBeenCalled();
  });

  it("can be dismissed without reloading", async () => {
    mockGateway();

    const client = renderWithClient(<UpdatePrompt />);
    await waitForLoadedBuild(client, "build-a");

    servedBuild = "build-b";
    await act(() => client.refetchQueries({ queryKey: ["build"] }));
    await userEvent.click(await screen.findByRole("button", { name: "Later" }));

    expect(screen.queryByText(/An update is available/)).not.toBeInTheDocument();
  });

  it("says nothing when the build check fails", async () => {
    // A gateway that cannot answer must not put a banner in the operator's way.
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));

    renderWithClient(<UpdatePrompt />);
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());

    expect(screen.queryByText(/An update is available/)).not.toBeInTheDocument();
  });
});
