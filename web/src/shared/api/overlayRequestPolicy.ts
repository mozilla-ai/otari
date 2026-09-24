/**
 * Where the management API is, and what credential a request to it carries.
 *
 * This build answers "the page's own origin, with its own cookies", which is
 * every deployment that serves its dashboard beside its API. A build whose one
 * dashboard reaches several deployments replaces this module at build time and
 * answers with the origin the person chose; that version owns how the choice
 * is made and kept, so the base client holds no state for a topology it does
 * not have. The shape it answers with is `RequestPolicy` in `./requestPolicy`,
 * kept off this module so the replacing one can import it.
 *
 * `prepareRequests` runs once, before the bootstrap is read, so a replacing
 * module can settle its answer (a stored choice, a directory it has to fetch)
 * ahead of the first request that depends on it. It must resolve rather than
 * reject: `main.tsx` treats a rejection as "same origin" and carries on, since
 * a dashboard that cannot decide where its API is still has the origin it was
 * served from.
 *
 * **Reached by its `@/shared/api/overlayRequestPolicy` specifier and never
 * relatively**, which is the seam rule and not a style call; `overlaySeams.test.ts`
 * enforces it and web/AGENTS.md says why.
 */

import { type RequestPolicy, SAME_ORIGIN_POLICY } from "./requestPolicy"

export async function prepareRequests(): Promise<void> {}

export function requestPolicy(): RequestPolicy {
  return SAME_ORIGIN_POLICY
}
