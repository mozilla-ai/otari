import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { deleteSession, setUnauthorizedHandler } from "@/api/client";

// Non-secret marker that a session cookie was minted for this browser. The
// credential itself is an HttpOnly cookie the page cannot read, so this flag is
// what lets the app render signed-in synchronously on load instead of probing
// the server first. If it is ever stale (cookie expired or revoked), the first
// 401 drops it and bounces to sign-in, exactly like any mid-session revocation.
const STORAGE_KEY = "otari.dashboard.hasSession";

interface AuthContextValue {
  isAuthenticated: boolean;
  login: () => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function readStoredMarker(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();

  const [isAuthenticated, setAuthenticated] = useState<boolean>(readStoredMarker);

  const logout = useCallback(() => {
    // Best-effort server-side revocation; local sign-out proceeds regardless.
    void deleteSession();
    setAuthenticated(false);
    // Drop any admin data cached under the old session so it can't render to a
    // later, possibly different, session in the same tab.
    queryClient.clear();
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage errors (e.g. private mode); in-memory state still clears.
    }
  }, [queryClient]);

  // Called after POST /v1/auth/session succeeded, i.e. the browser already
  // holds the session cookie; this only flips the rendered state.
  const login = useCallback(() => {
    // Clear any cache from a prior session before the new session's queries run.
    queryClient.clear();
    setAuthenticated(true);
    try {
      window.localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      // Ignore storage errors; the sign-in still works for this tab.
    }
  }, [queryClient]);

  // A 401 from any request means the session expired or was revoked: drop it.
  useEffect(() => {
    setUnauthorizedHandler(logout);
    return () => setUnauthorizedHandler(null);
  }, [logout]);

  const value = useMemo<AuthContextValue>(
    () => ({ isAuthenticated, login, logout }),
    [isAuthenticated, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
