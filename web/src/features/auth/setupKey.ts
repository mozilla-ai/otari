/**
 * The master key a first-run setup link carries: `#/setup?key=…`, printed once
 * in the gateway's log (`services/master_key_service.py`).
 *
 * The key is in the fragment so it reaches no server, access log or `Referer`
 * header. It is taken out of the address bar as soon as it is read, so it is
 * not left in the tab's history or in a copied URL, and held here in memory
 * only: a reload forgets it, and the link in the log still works until the
 * password it sets retires the key as a sign-in.
 */
let setupKey: string | null = null

/**
 * The setup key in `hash` or one captured earlier, or null when there is none.
 * A key found in `hash` is removed from the address bar.
 *
 * `main.tsx` calls this before the bootstrap is fetched, and the setup page
 * calls it again with the hash it was rendered for, which covers a link pasted
 * into a tab that is already open.
 */
export function captureSetupKey(hash: string): string | null {
  const [path, query] = hash.replace(/^#/, "").split("?")
  if (path !== "/setup") {
    return setupKey
  }
  const key = new URLSearchParams(query ?? "").get("key")
  if (key) {
    setupKey = key
    window.history.replaceState(window.history.state, "", "#/setup")
  }
  return setupKey
}

/** Drop the captured key, once it has been spent or refused. */
export function forgetSetupKey(): void {
  setupKey = null
}
