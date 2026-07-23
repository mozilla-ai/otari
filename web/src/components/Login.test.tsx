import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useAuth } from "@/auth/AuthContext";
import { Login } from "@/components/Login";
import { Provider } from "@/provider";

function Harness() {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <div>SIGNED IN</div> : <Login />;
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
      <Provider>
        <Harness />
      </Provider>,
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
      <Provider>
        <Harness />
      </Provider>,
    );

    const link = screen.getByRole("link", { name: /welcome/i });
    expect(link).toHaveAttribute("href", "/welcome");
  });

  it("shows an error and stays on the form when the key is rejected", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ detail: "Invalid master key" }, 401));
    const user = userEvent.setup();

    render(
      <Provider>
        <Harness />
      </Provider>,
    );

    await user.type(screen.getByLabelText("Master key"), "sk-wrong");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Invalid master key.")).toBeInTheDocument();
    expect(screen.queryByText("SIGNED IN")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeInTheDocument();
  });
});
