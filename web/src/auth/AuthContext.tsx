import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { setMasterKey, setUnauthorizedHandler } from "@/api/client";

const STORAGE_KEY = "otari.dashboard.masterKey";

interface AuthContextValue {
  masterKey: string | null;
  isAuthenticated: boolean;
  login: (key: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function readStoredKey(): string | null {
  try {
    return window.sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();

  // Seed the api client synchronously during the first render so a restored
  // session key is in place before any child query fires. Doing this in an
  // effect would let the first request go out unauthenticated (effects run
  // child-first, so React Query's fetch would race ahead of the sync).
  const [masterKey, setKey] = useState<string | null>(() => {
    const stored = readStoredKey();
    setMasterKey(stored);
    return stored;
  });

  const logout = useCallback(() => {
    setMasterKey(null);
    setKey(null);
    // Drop any admin data cached under the old key so it can't render to a
    // later, possibly different, session in the same tab.
    queryClient.clear();
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage errors (e.g. private mode); in-memory state still clears.
    }
  }, [queryClient]);

  const login = useCallback(
    (key: string) => {
      const trimmed = key.trim();
      // Set the client key synchronously (before the re-render that mounts the
      // dashboard) so the first authenticated request carries the header.
      setMasterKey(trimmed);
      // Clear any cache from a prior session before the new key's queries run.
      queryClient.clear();
      setKey(trimmed);
      try {
        window.sessionStorage.setItem(STORAGE_KEY, trimmed);
      } catch {
        // Ignore storage errors; the key still lives in memory for this session.
      }
    },
    [queryClient],
  );

  // A 401 from any request means the key is wrong or was revoked: drop it.
  useEffect(() => {
    setUnauthorizedHandler(logout);
    return () => setUnauthorizedHandler(null);
  }, [logout]);

  const value = useMemo<AuthContextValue>(
    () => ({ masterKey, isAuthenticated: masterKey != null, login, logout }),
    [masterKey, login, logout],
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
