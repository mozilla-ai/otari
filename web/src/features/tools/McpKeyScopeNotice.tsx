import { Link } from "@tanstack/react-router"

import type { ApiKey } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { SCOPE_INFO } from "@/features/keys/scopes"
import { useKeys, useKeysScope } from "@/shared/api/apiKeys"

/** A key that can still authenticate: switched on and not past its expiry. */
function isUsable(key: ApiKey, now: number): boolean {
  if (!key.is_active) return false
  if (!key.expires_at) return true
  const expires = Date.parse(key.expires_at)
  return Number.isNaN(expires) || expires > now
}

/** "1 of your 3 keys in this workspace can call them." and its variants. */
export function holdersSentence(
  holders: number,
  total: number,
  isDeploymentWide: boolean,
): string {
  const keys = `${total} ${total === 1 ? "key" : "keys"}`
  const whose = isDeploymentWide
    ? `the ${keys} in this workspace`
    : `your ${keys} in this workspace`
  if (holders === 0 && total === 1) {
    return `${isDeploymentWide ? "The one" : "Your"} key in this workspace cannot call them yet.`
  }
  if (holders === 0) return `None of ${whose} can call them yet.`
  return `${holders} of ${whose} can call them.`
}

/**
 * Which keys can use the workspace's MCP servers, for the page that lists them.
 *
 * A key may be created without `mcp:execute`, and a server reaches a request only
 * when the key holds `mcp:execute`, so a server added for a fresh key silently
 * does nothing. This says so where that is noticed, and links to the keys page
 * rather than offering to widen every key from here.
 *
 * The count covers the keys this caller can read: every key in the workspace
 * for a deployment operator, otherwise their own and, for an organization
 * owner or admin, its service keys. It is gone once every one of them holds the
 * scope, and says nothing until the keys are read, so it does not flash.
 */
export function McpKeyScopeNotice({ workspaceId }: { workspaceId: string }) {
  const scope = useKeysScope()
  const keys = useKeys(workspaceId)
  if (keys.isPending) return null

  // A failed refresh keeps the last answer in `keys.data`; a count from it
  // could be out of date, so a failed read of any kind shows no count.
  const read = keys.isError ? undefined : keys.data
  const usable = (read ?? []).filter((key) => isUsable(key, Date.now()))
  const holders = usable.filter((key) =>
    key.scopes.includes("mcp:execute"),
  ).length
  const isCounted = usable.length > 0
  if (isCounted && holders === usable.length) return null

  const isWarning = isCounted && holders === 0

  return (
    <InfoBanner tone={isWarning ? "warning" : "info"}>
      <p>
        A request reaches these servers only through a key with{" "}
        <span className="text-foreground">
          {SCOPE_INFO["mcp:execute"].label}
        </span>{" "}
        (<code>mcp:execute</code>). A client that lists them first, for a
        server's id or its sign-in link, also needs{" "}
        <span className="text-foreground">{SCOPE_INFO["mcp:read"].label}</span>{" "}
        (<code>mcp:read</code>).
        {isCounted
          ? ` ${holdersSentence(holders, usable.length, scope.isDeploymentWide)}`
          : null}{" "}
        Change a key's access on{" "}
        <Link to="/keys" className="text-link hover:text-link-hover">
          API keys
        </Link>
        .
      </p>
    </InfoBanner>
  )
}
