import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { GatewaySettings } from "@/api/types";
import { PricingWarning } from "@/components/PricingWarning";

const BASE: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  model_discovery: true,
  default_pricing: false,
  require_pricing: true,
  master_key_source: "configured",
  config: [],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function mockSettings(settings: GatewaySettings) {
  let current = { ...settings };
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (url.includes("/v1/settings")) {
      if (method === "PATCH") current = { ...current, ...JSON.parse(String(init?.body)) };
      return jsonResponse(current);
    }
    return jsonResponse({});
  });
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("PricingWarning", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("alarms and enables default pricing when require_pricing rejects requests", async () => {
    const fetchMock = mockSettings({ ...BASE, require_pricing: true, default_pricing: false });
    const user = userEvent.setup();
    renderPage(<PricingWarning />);

    const enable = await screen.findByRole("button", { name: "Enable default pricing" });
    await user.click(enable);

    const patch = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/settings") && (init?.method ?? "") === "PATCH",
    );
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ default_pricing: true });
    // Once default pricing is on, the alarm clears.
    await waitFor(() => expect(screen.queryByText(/Requests are rejected/)).not.toBeInTheDocument());
  });

  it("stays quiet when default pricing is already on", async () => {
    mockSettings({ ...BASE, require_pricing: true, default_pricing: true });
    renderPage(<PricingWarning />);

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    expect(screen.queryByText(/Requests are rejected/)).not.toBeInTheDocument();
  });

  it("can be dismissed", async () => {
    mockSettings({ ...BASE, require_pricing: true, default_pricing: false });
    const user = userEvent.setup();
    renderPage(<PricingWarning />);

    await user.click(await screen.findByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(/Requests are rejected/)).not.toBeInTheDocument();
  });
});
