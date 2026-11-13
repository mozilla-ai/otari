import { ApiError } from "@/shared/api/client"

// A UUID anywhere in a message means it was written for an API caller, not for
// the person on this page.
const UUID = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i

/**
 * One plain line for a failed MCP server action (a check, a sign-in), never the
 * raw detail of an error written for an API caller. The gateway's own sentences
 * pass through where they are already plain; anything naming an id, or not
 * from the gateway at all, gets `fallback`.
 */
export function mcpErrorMessage(error: unknown, fallback: string): string {
  if (!(error instanceof ApiError)) return fallback
  if (error.status === 429) {
    return "Too many checks are running right now. Try again in a moment."
  }
  if (error.status === 404) {
    return "This server was deleted, or you can no longer see it."
  }
  if (error.status === 403 || error.status === 401) {
    return "You are not allowed to do this here any more."
  }
  if (error.status >= 500 || UUID.test(error.message) || !error.message) {
    return fallback
  }
  return error.message
}

/**
 * Whether a refusal asks for an OAuth client ID, which an admin enters in the
 * server's Settings. The gateway says so in these words wherever a sign-in
 * cannot start without one.
 */
export function asksForOAuthClient(message: string): boolean {
  return /OAuth client ID/i.test(message)
}
