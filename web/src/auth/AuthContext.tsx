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
  const [masterKey, setKey] = useState<string | null>(() => readStoredKey());

  // Keep the api client's in-memory key in sync with React state.
  useEffect(() => {
    setMasterKey(masterKey);
  }, [masterKey]);

  const logout = useCallback(() => {
    setKey(null);
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage errors (e.g. private mode); in-memory state still clears.
    }
  }, []);

  const login = useCallback((key: string) => {
    const trimmed = key.trim();
    setKey(trimmed);
    try {
      window.sessionStorage.setItem(STORAGE_KEY, trimmed);
    } catch {
      // Ignore storage errors; the key still lives in memory for this session.
    }
  }, []);

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
