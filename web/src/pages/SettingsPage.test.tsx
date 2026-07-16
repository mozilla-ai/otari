import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { GatewaySettings } from "@/api/types";
import { SettingsPage } from "@/pages/SettingsPage";

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.2.3",
  model_discovery: true,
  default_pricing: false,
  require_pricing: false,
};

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

// Serves the settings, and echoes PATCH bodies back merged onto the base so the
// UI reflects the toggle after a change.
function mockApi(initial: GatewaySettings = SETTINGS) {
  let current = initial;
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase();
    if (String(input).includes("/v1/settings")) {
      if (method === "PATCH") {
        current = { ...current, ...JSON.parse(String(init?.body)) };
      }
      return jsonResponse(current);
    }
    return jsonResponse([]);
  });
}

describe("SettingsPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("reflects the current settings on its switches", async () => {
    mockApi();

    renderWithClient(<SettingsPage />);

    // Wait for the settings to load (the switch renders unchecked while loading).
    await screen.findByText(/Version 1.2.3/);
    expect(screen.getByRole("switch", { name: "Model discovery" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("switch", { name: "Default pricing" })).toHaveAttribute("aria-checked", "false");
  });

  it("patches a setting when its switch is toggled", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByRole("switch", { name: "Model discovery" });

    await user.click(screen.getByRole("switch", { name: "Model discovery" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(call).toBeDefined();
    expect(String(call?.[0])).toContain("/v1/settings");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ model_discovery: false });

    // The switch reflects the new value after the round trip.
    expect(await screen.findByRole("switch", { name: "Model discovery" })).toHaveAttribute("aria-checked", "false");
  });

  it("enables default pricing from off", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByRole("switch", { name: "Default pricing" });

    await user.click(screen.getByRole("switch", { name: "Default pricing" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ default_pricing: true });
  });
});
