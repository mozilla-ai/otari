import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { CreateKeyResponse, KeyInfo } from "@/api/types";
import { KeysPage } from "@/pages/KeysPage";

const SAMPLE_KEY: KeyInfo = {
  id: "key-1",
  key_name: "ci-pipeline",
  user_id: "apikey-123",
  created_at: "2026-01-01T10:00:00Z",
  last_used_at: null,
  expires_at: null,
  is_active: true,
  metadata: {},
};

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("KeysPage", () => {
  beforeEach(() => {
    setMasterKey("test-master-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("lists existing keys returned by the gateway", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse([SAMPLE_KEY]));

    renderWithClient(<KeysPage />);

    expect(await screen.findByText("ci-pipeline")).toBeInTheDocument();
    expect(screen.getByText("active")).toBeInTheDocument();
  });

  it("creates a key and surfaces the one-time secret", async () => {
    const created: CreateKeyResponse = {
      id: "key-2",
      key: "sk-secret-value",
      key_name: "new-key",
      user_id: "apikey-456",
      created_at: "2026-01-02T10:00:00Z",
      expires_at: null,
      is_active: true,
      metadata: {},
    };

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      const method = (init?.method ?? "GET").toUpperCase();
      if (method === "POST") {
        return jsonResponse(created);
      }
      return jsonResponse([SAMPLE_KEY]);
    });

    const user = userEvent.setup();
    renderWithClient(<KeysPage />);

    await screen.findByText("ci-pipeline");

    await user.click(screen.getByRole("button", { name: "Create key" }));
    // The header toggle now reads "Hide form", so the only "Create key" button
    // left is the form's submit.
    await screen.findByRole("heading", { name: "Create API key" });
    await user.click(screen.getByRole("button", { name: "Create key" }));

    expect(await screen.findByText("sk-secret-value")).toBeInTheDocument();
    const postCalls = fetchMock.mock.calls.filter(([, init]) => (init?.method ?? "GET").toUpperCase() === "POST");
    expect(postCalls).toHaveLength(1);
    expect(postCalls[0][0]).toBe("/v1/keys");
  });
});
