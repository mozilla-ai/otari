// Which model this browser was last using, per workspace.
//
// Browser storage rather than the server, and the distinction is not arbitrary:
// pinned models are a preference somebody set and expects on their other
// devices, so those are stored; "the model I had open" is where this tab left
// off, which is per browser by nature. It is also the only thing on this page
// that can be lost with no consequence, which is why every access is
// best-effort.

const KEY_PREFIX = "otari.playground.model."

function storageKey(workspaceId: string): string {
  return `${KEY_PREFIX}${workspaceId}`
}

/** The workspace's last-used model key, or undefined when there is none. */
export function readRememberedModel(workspaceId: string): string | undefined {
  try {
    return window.localStorage.getItem(storageKey(workspaceId)) ?? undefined
  } catch {
    // A private window and a blocked-storage policy both throw here. The
    // fallback is the catalog's first model, which is a working Playground.
    return undefined
  }
}

/** Remember the workspace's chosen model. Best-effort. */
export function rememberModel(workspaceId: string, modelKey: string): void {
  try {
    window.localStorage.setItem(storageKey(workspaceId), modelKey)
  } catch {
    // Quota or a blocked policy. Nothing to recover: the next visit starts on
    // the catalog's first model.
  }
}
