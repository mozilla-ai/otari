import { useQueryClient } from "@tanstack/react-query"
import type { ReactNode } from "react"
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"

import { deleteSession, setUnauthorizedHandler } from "@/shared/api/client"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

// Non-secret marker that a session cookie was minted for this browser. The
// credential itself is an HttpOnly cookie the page cannot read, so this flag is
// what lets the app render signed-in synchronously on load instead of probing
// the server first. If it is ever stale (cookie expired or revoked), the first
// 401 drops it and bounces to sign-in, exactly like any mid-session revocation.
const STORAGE_KEY = "otari.dashboard.hasSession"

interface AuthContextValue {
  isAuthenticated: boolean
  // True while a sign-out's server-side revocation is still in flight. The
  // sign-in screen uses this to refuse a new credential until the old
  // session has finished tearing down, so a stalled DELETE cannot outlive a
  // fresh sign-in and clobber its cookie with this call's expiring one
  // (see #557).
  isSigningOut: boolean
  login: () => void
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

function readStoredMarker(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1"
  } catch {
    return false
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const { recordEvent, identify } = useTelemetry()

  const [isAuthenticated, setAuthenticated] =
    useState<boolean>(readStoredMarker)
  const [isSigningOut, setSigningOut] = useState(false)
  // logout() can fire more than once concurrently: a manual sign-out and a
  // stray 401-triggered auto-logout (unauthorizedHandler) can both land
  // close together, each starting its own deleteSession(). Counting the
  // in-flight revocations, rather than a single finally() clearing the flag
  // unconditionally, means isSigningOut only drops once every pending one has
  // settled - not just whichever happens to resolve first.
  const pendingSignOutsRef = useRef(0)

  const logout = useCallback(() => {
    // Recorded before anything is torn down, and from here rather than from the
    // account menu, because this is also the path a 401 takes: a session that
    // expired or was revoked ends the same funnel a deliberate sign-out does.
    // `identify(null)` is the other half of it, and it is what stops the next
    // session in this tab from being attributed to the identity that just left.
    recordEvent(TELEMETRY_EVENTS.LOGOUT)
    identify(null)
    // Local sign-out is unconditional and synchronous, exactly as before:
    // the UI returns to the sign-in screen at once regardless of how the
    // server-side revocation below turns out.
    setAuthenticated(false)
    // Drop any admin data cached under the old session so it can't render to a
    // later, possibly different, session in the same tab.
    queryClient.clear()
    try {
      window.localStorage.removeItem(STORAGE_KEY)
    } catch {
      // Ignore storage errors (e.g. private mode); in-memory state still clears.
    }
    // Best-effort server-side revocation, now bounded (see client.ts) and
    // tracked: isSigningOut gates the sign-in form so a new session cannot
    // be minted while any revocation might still land and clear its cookie.
    pendingSignOutsRef.current += 1
    setSigningOut(true)
    void deleteSession().finally(() => {
      pendingSignOutsRef.current -= 1
      if (pendingSignOutsRef.current === 0) {
        setSigningOut(false)
      }
    })
  }, [queryClient, recordEvent, identify])

  // Called after POST /v1/auth/session succeeded, i.e. the browser already
  // holds the session cookie; this only flips the rendered state.
  const login = useCallback(() => {
    // Clear any cache from a prior session before the new session's queries run.
    queryClient.clear()
    setAuthenticated(true)
    try {
      window.localStorage.setItem(STORAGE_KEY, "1")
    } catch {
      // Ignore storage errors; the sign-in still works for this tab.
    }
  }, [queryClient])

  // A 401 from any request means the session expired or was revoked: drop it.
  useEffect(() => {
    setUnauthorizedHandler(logout)
    return () => setUnauthorizedHandler(null)
  }, [logout])

  const value = useMemo<AuthContextValue>(
    () => ({ isAuthenticated, isSigningOut, login, logout }),
    [isAuthenticated, isSigningOut, login, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return ctx
}
