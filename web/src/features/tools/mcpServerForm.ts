import type { McpCredentialMode } from "@/client"
import type { RadioOption } from "@/design-system/forms/RadioGroup"

// The form rules the add dialog and a server's Settings dialog share.

/**
 * Comma separated in a form, a list on the wire, and null rather than `[]` for
 * "expose every tool this server offers": the tool loop reads `[]` as a
 * deny-all, so a blank field must not send one.
 */
export function parseAllowedTools(raw: string): string[] | null {
  const names = raw
    .split(",")
    .map((name) => name.trim())
    .filter((name) => name !== "")
  return names.length > 0 ? names : null
}

/**
 * The same two rules the gateway applies when it stores a URL
 * (`services/url_safety.validate_mcp_url`): an http(s) endpoint, and https once
 * a bearer token rides on it. Checked here as well so the message lands under
 * the field rather than arriving as a banner after a round trip. The SSRF half
 * is deliberately not mirrored: it resolves the host, which only the gateway
 * can do, and it stays the authority either way.
 */
export function urlProblem(
  raw: string,
  willHaveToken: boolean,
  isPerUser = false,
): string | undefined {
  const trimmed = raw.trim()
  if (trimmed === "") return undefined
  let parsed: URL
  try {
    parsed = new URL(trimmed)
  } catch {
    return "Give the server's full endpoint, such as https://mcp.example.com/github."
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return "An MCP endpoint is reached over http or https."
  }
  if (isPerUser && parsed.protocol !== "https:") {
    return "A personal server needs an https URL, so each member's own sign-in is not sent in the clear."
  }
  if (willHaveToken && parsed.protocol !== "https:") {
    return "A server with an authorization token needs an https URL, so the token is not sent in the clear."
  }
  return undefined
}

/**
 * What the dashboard calls a server's credential mode. UI words only: the API
 * keeps `shared` and `per_user`.
 */
export const CREDENTIAL_MODE_COPY: Record<
  McpCredentialMode,
  { label: string; longLabel: string; help: string }
> = {
  shared: {
    label: "Shared",
    longLabel: "Shared account",
    help: "One account for the whole workspace. An admin signs in for everyone, or pastes a token.",
  },
  per_user: {
    label: "Personal",
    longLabel: "Personal accounts",
    help: "Each member connects their own account.",
  },
}

/** The label of the control that picks a mode, in every form that has one. */
export const SIGN_IN_MODE_LABEL = "Sign-in mode"

/** The two modes as every form offers them, spelled out in full. */
export const SIGN_IN_MODES: RadioOption[] = (
  ["shared", "per_user"] as const
).map((value) => ({
  value,
  label: CREDENTIAL_MODE_COPY[value].longLabel,
  description: CREDENTIAL_MODE_COPY[value].help,
}))
