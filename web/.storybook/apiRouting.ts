import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"

/**
 * Where a request made inside the catalog should go, decided without touching
 * the network.
 *
 * Split out of `apiMock.tsx` so it can be tested: that file replaces
 * `globalThis.fetch` at module scope, so importing it from a test would patch
 * the test process's own fetch as a side effect of asking a question about a
 * path. Nothing here has a side effect.
 */

/**
 * The gateway's prefix. Everything else belongs to Storybook itself.
 *
 * Derived from `API_ROOT` rather than spelled again: a second literal is how
 * this broke. The API moved to `/api/v1` and the mock kept matching `/v1/`, so
 * every story's request missed the mock and went to the network, where the
 * static catalog answers 404. The constant's own comment promises that moving
 * the API is "a change here and nowhere else", which only holds if nothing
 * else writes the path down.
 */
const GATEWAY_PREFIX = `${API_ROOT}/`

export interface MockFailure {
  $status: number
  $body?: unknown
}

export type ApiMocks = Record<string, unknown>

export type Routed =
  | { kind: "network" }
  | { kind: "response"; status: number; body: unknown }

function isMockFailure(value: unknown): value is MockFailure {
  return typeof value === "object" && value !== null && "$status" in value
}

/**
 * The gateway a story gets without asking, for what the decorators mount around
 * it rather than what it declared.
 *
 * `withAppContext` wraps every story in `SelectedWorkspaceProvider`, which seeds
 * itself from `useOrganizationContext()`. So a story about a `Divider` queries
 * the organization, and before this existed that request found no table, fell
 * through to the network, and 404d against whatever origin served the catalog.
 * Every story in the published catalog did it, and the smoke run reported them
 * clean because it dropped the whole "Failed to load resource" family.
 *
 * A story that cares overrides any of this through `parameters.api`, which is
 * checked first. Keep this to what a decorator causes: a path only one component
 * needs belongs in that component's story, where a reader can see it.
 */
export const BASELINE: ApiMocks = {
  [`${API_ROOT}/organizations/me`]: organizationContext(),
}

/**
 * The request's path and query, whatever shape the caller passed.
 *
 * `apiFetch` passes a relative string, which is why this used to return the
 * three cases as they came. It normalizes now because `route` keys on the
 * leading `API_ROOT`, and a `Request` carries an absolute URL: left alone, the
 * same path would be mocked as a string and reach the network as a `Request`.
 */
export function pathOf(input: RequestInfo | URL): string {
  const raw =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url
  try {
    const url = new URL(raw, globalThis.location?.origin ?? "http://localhost")
    return url.pathname + url.search
  } catch {
    return raw
  }
}

/** Match the whole path first, then its pathname. */
function lookup(mocks: ApiMocks, path: string): unknown {
  if (path in mocks) return mocks[path]
  const pathname = path.split("?")[0]
  return pathname in mocks ? mocks[pathname] : undefined
}

function responseFor(match: unknown): Routed {
  return isMockFailure(match)
    ? { kind: "response", status: match.$status, body: match.$body }
    : { kind: "response", status: 200, body: match }
}

/**
 * `tables` is the mock table of every story currently on screen, most recent
 * first is not assumed: an autodocs page mounts several at once and the first
 * one carrying the path wins, which is the same order the live set iterates.
 */
export function route(path: string, tables: Iterable<ApiMocks>): Routed {
  // Not the gateway, so it is Storybook's own: the story index, a bundle chunk,
  // a font. These have to reach the network or the catalog does not load.
  if (!path.startsWith(GATEWAY_PREFIX)) return { kind: "network" }

  for (const mocks of tables) {
    const match = lookup(mocks, path)
    if (match !== undefined) return responseFor(match)
  }

  const baseline = lookup(BASELINE, path)
  if (baseline !== undefined) return responseFor(baseline)

  // 501, not 404: a 404 is a real gateway answer that some components handle
  // gracefully, which would hide the fact that a story forgot a path.
  return {
    kind: "response",
    status: 501,
    body: { detail: `No story mock for ${path}. Add it to parameters.api.` },
  }
}
