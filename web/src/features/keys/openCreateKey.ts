import { createContext, useContext } from "react"

/** Where the Create API key dialog was opened from, as recorded. */
export type CreateKeyEntryPoint =
  | "sidebar"
  | "api_keys_page"
  | "api_keys_empty"
  | "member_first_run"

/**
 * Opens the shell's Create API key dialog from any page, naming the control
 * that asked and, when that control will be gone by the time the dialog
 * closes, where focus goes instead (the drawer's toggle, for the drawer's own
 * Create API key on a phone).
 */
export type OpenCreateKey = (
  entryPoint: CreateKeyEntryPoint,
  returnFocusTo?: HTMLElement | null,
) => void

export const OpenCreateKeyContext = createContext<OpenCreateKey>(() => {})

export function useOpenCreateKey(): OpenCreateKey {
  return useContext(OpenCreateKeyContext)
}
