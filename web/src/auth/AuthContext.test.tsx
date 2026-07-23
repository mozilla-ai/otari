import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useAuth } from "@/auth/AuthContext";
import { Provider } from "@/provider";

function Harness() {
  const { isAuthenticated, logout } = useAuth();
  return isAuthenticated ? (
    <button type="button" onClick={logout}>
      Sign out
    </button>
  ) : (
    <div>SIGNED OUT</div>
  );
}

describe("AuthProvider", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("restores the signed-in state on load without asking for the key again", () => {
    // The session credential is an HttpOnly cookie the page cannot read; the
    // persisted marker is what tells a fresh tab/restart it is signed in.
    window.localStorage.setItem("otari.dashboard.hasSession", "1");

    render(
      <Provider>
        <Harness />
      </Provider>,
    );

    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
  });

  it("starts signed out when no session marker is present", () => {
    render(
      <Provider>
        <Harness />
      </Provider>,
    );

    expect(screen.getByText("SIGNED OUT")).toBeInTheDocument();
  });

  it("revokes the server-side session and drops the marker on sign-out", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 204 }));
    const user = userEvent.setup();

    render(
      <Provider>
        <Harness />
      </Provider>,
    );

    await user.click(screen.getByRole("button", { name: "Sign out" }));

    expect(screen.getByText("SIGNED OUT")).toBeInTheDocument();
    expect(window.localStorage.getItem("otari.dashboard.hasSession")).toBeNull();
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([url]) => url === "/v1/auth/session");
      expect(call?.[1]?.method).toBe("DELETE");
    });
  });
});
