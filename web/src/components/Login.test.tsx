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
    window.sessionStorage.clear();
  });

  it("signs in when the master key is accepted and sends it as a Bearer token", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse([]));
    const user = userEvent.setup();

    render(
      <Provider>
        <Harness />
      </Provider>,
    );

    await user.type(screen.getByLabelText("Master key"), "sk-correct");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument();

    const [, init] = fetchMock.mock.calls[0];
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer sk-correct");
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
