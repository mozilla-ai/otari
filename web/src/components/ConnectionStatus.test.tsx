import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "@/api/client";
import { ConnectionStatus } from "@/components/ConnectionStatus";

// Drives one management request so the query cache carries a real success/error,
// exactly what ConnectionStatus watches.
function Probe() {
  useQuery({ queryKey: ["probe"], queryFn: () => apiFetch("/v1/settings"), retry: false });
  return null;
}

function renderWithProbe(): { client: QueryClient } {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrap = (children: ReactNode) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  render(
    wrap(
      <>
        <Probe />
        <ConnectionStatus />
      </>,
    ),
  );
  return { client };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("ConnectionStatus", () => {
  afterEach(() => vi.restoreAllMocks());

  it("alerts when the gateway is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    renderWithProbe();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Can’t reach the gateway/);
  });

  it("clears itself once the gateway responds again", async () => {
    let online = false;
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      if (!online) throw new TypeError("Failed to fetch");
      return jsonResponse({ ok: true });
    });
    const { client } = renderWithProbe();

    await screen.findByRole("alert");

    online = true;
    await client.refetchQueries({ queryKey: ["probe"] });

    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("stays quiet for an error that is not a lost connection", async () => {
    // A 500 means the backend answered; that is a page-level error, not "offline".
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ detail: "boom" }, 500));
    renderWithProbe();

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });
});
