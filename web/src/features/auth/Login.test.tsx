import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useAuth } from "@/features/auth/AuthContext";
import { Login } from "@/features/auth/Login";
import { AppProviders } from "@/tests/providers";

function Harness() {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <div>SIGNED IN</div> : <Login />;
}

function SignOutThenLoginHarness() {
  const { isAuthenticated, logout } = useAuth();
  return isAuthenticated ? (
    <button type="button" onClick={logout}>
      Sign out
    </button>
  ) : (
    <Login />
  );
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("Login", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("signs in by exchanging the master key for a session, never storing the key", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }));
    const user = userEvent.setup();

    render(
      <AppProviders>
        <Harness />
      </AppProviders>,
    );

    await user.type(screen.getByLabelText("Master key"), "sk-correct");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument();

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/v1/auth/session");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ master_key: "sk-correct" }));
    // The raw key must not land in any JS-readable storage.
    expect(window.localStorage.getItem("otari.dashboard.hasSession")).toBe("1");
    expect(Object.values({ ...window.localStorage })).not.toContain("sk-correct");
    expect(Object.values({ ...window.sessionStorage })).not.toContain("sk-correct");
  });

  it("links to the auth-free welcome page", () => {
    render(
      <AppProviders>
        <Harness />
      </AppProviders>,
    );

    const link = screen.getByRole("link", { name: /welcome/i });
    expect(link).toHaveAttribute("href", "/welcome");
  });

  it("shows an error and stays on the form when the key is rejected", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ detail: "Invalid master key" }, 401));
    const user = userEvent.setup();

    render(
      <AppProviders>
        <Harness />
      </AppProviders>,
    );

    await user.type(screen.getByLabelText("Master key"), "sk-wrong");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Invalid master key.")).toBeInTheDocument();
    expect(screen.queryByText("SIGNED IN")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeInTheDocument();
  });

  it("refuses a new credential while a prior sign-out's revocation is still in flight (#557)", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1");

    let resolveDelete!: () => void;
    const deletePending = new Promise<Response>((resolve) => {
      resolveDelete = () => resolve(new Response(null, { status: 204 }));
    });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      if (init?.method === "DELETE") {
        return deletePending;
      }
      return Promise.resolve(jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }));
    });
    const user = userEvent.setup();

    render(
      <AppProviders>
        <SignOutThenLoginHarness />
      </AppProviders>,
    );

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    // Local sign-out lands immediately, before the DELETE resolves.
    const keyField = await screen.findByLabelText("Master key");
    await user.type(keyField, "sk-new");

    const submitButton = screen.getByRole("button", { name: "Finishing sign-out…" });
    expect(submitButton).toBeDisabled();
    await user.click(submitButton);

    // Blocked: no sign-in POST was attempted while the old sign-out was pending.
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);

    resolveDelete();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled();
    });

    await user.click(screen.getByRole("button", { name: "Sign in" }));

    // Back to authenticated: SignOutThenLoginHarness renders the "Sign out"
    // button again once isAuthenticated flips true.
    expect(await screen.findByRole("button", { name: "Sign out" })).toBeInTheDocument();
    const postCall = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
    expect(postCall?.[1]?.body).toBe(JSON.stringify({ master_key: "sk-new" }));
  });
});
