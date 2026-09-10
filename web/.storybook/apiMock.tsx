import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import type { Decorator } from "@storybook/react-vite"

import { type ApiMocks, pathOf, route } from "./apiRouting"

/**
 * A stub gateway for the stories whose component fetches its own data.
 *
 * Most of the catalog is prop-driven and needs none of this. A handful of feature
 * components own their queries (`MailDeliveryCard`, `SetupGuideCard`,
 * `ModelComboBox`, ...), and the house rule for those is the one in
 * `.github/skills/frontend-standards/testing.md`: mock the network boundary,
 * nothing inside it. There is no `vi.mock("@/shared/api/hooks")` anywhere in the
 * tree and this does not become the first -- it replaces `fetch`, which is what
 * `apiFetch` sits on, so the component's real hooks, real query keys and real
 * error handling all run.
 *
 * Deliberately not MSW: `msw-storybook-addon` wants `mockServiceWorker.js` in
 * `web/public/`, which the gateway serves in production, and it would be a third
 * tracked file to keep out of a commit. This is smaller than the plumbing to
 * avoid that.
 *
 * A story declares its gateway as a parameter, keyed by path. Values are usually
 * builders from `src/tests/fixtures.ts` -- reuse those rather than hand-writing an
 * API shape, so a field the gateway adds arrives here too:
 *
 *   parameters: {
 *     api: {
 *       [`${API_ROOT}/settings/mail`]: { configured: true, from_address: "otari@example.com" },
 *       [`${API_ROOT}/organizations/me`]: organizationContext(),
 *     },
 *   }
 *
 * To exercise a failure, give the path a `$status` envelope instead of a body:
 *
 *   api: { [`${API_ROOT}/settings/mail`]: { $status: 503, $body: { detail: "No transport." } } }
 *
 * The `$` prefix is what makes that unambiguous, and it is worth the ugliness. An
 * earlier version guessed instead -- "has a `status` or `body` key, therefore it is
 * an envelope" -- which is wrong here in a way that is very hard to read back:
 * plenty of the gateway's own DTOs carry a `status` field of their own
 * (`organizationContext`, `organizationMember`, `workspaceMember` and
 * `activationAttempt` all do), so an ordinary body was taken apart as an envelope
 * and served as `new Response(null, { status: "active" })`. The story rendered
 * empty with nothing in the console to say why.
 *
 * A `$`-prefixed key cannot collide with a generated DTO, and unlike a tagged
 * helper it needs no import -- which matters, because a story lives under `src/`
 * and importing out of this directory would pull it into the app's tsconfig.
 */
/**
 * The mock tables of every story currently on screen.
 *
 * A set rather than one "active" table, because an autodocs page mounts several
 * stories at once. React interleaves their renders, so a single slot would be
 * holding the last-rendered story's table by the time an earlier story's query
 * actually fired, and that story would render another one's data. Keyed by table
 * identity so a story can deregister exactly its own on unmount.
 */
const mounted = new Set<ApiMocks>()

// Captured at import, once, before anything below has replaced it. Reading it
// later would read the stub.
const realFetch: typeof fetch = globalThis.fetch.bind(globalThis)

function jsonResponse(status: number, body: unknown): Response {
  if (status === 204) return new Response(null, { status })
  return new Response(JSON.stringify(body ?? null), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// Installed once at module scope, not per story. Patching inside the decorator
// would reassign on every render, and the second pass would capture the stub as
// the "real" fetch and never be able to restore it.
//
// The decision itself is in `apiRouting.ts`, which has no side effects and is
// unit-tested; this is only the part that has to touch `globalThis`.
globalThis.fetch = async (input, init) => {
  const routed = route(pathOf(input), mounted)
  if (routed.kind === "network") return realFetch(input, init)
  return jsonResponse(routed.status, routed.body)
}

export const withApiMocks: Decorator = (Story, context) => {
  const mocks = (context.parameters.api ?? undefined) as ApiMocks | undefined

  // One client per story, created once. A module-level client would carry a
  // resolved query from the story you just looked at into the next one, which
  // reads as a component rendering stale data rather than as a leak.
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // No retries and no refetch: a catalog wants the state the story
            // declared, immediately, and an unmocked path should look obviously
            // broken rather than take three attempts to say so.
            retry: false,
            refetchOnWindowFocus: false,
            staleTime: Number.POSITIVE_INFINITY,
          },
          mutations: { retry: false },
        },
      }),
  )

  // Registered during render, not in an effect: a component's query fires from
  // its own first render, inside <Story /> below, which is before any effect here
  // would run. Adding to a set is idempotent, so StrictMode's double render and
  // the React Compiler are both free to repeat it. The effect exists only to
  // deregister.
  if (mocks) mounted.add(mocks)
  useEffect(() => {
    if (!mocks) return
    mounted.add(mocks)
    return () => {
      mounted.delete(mocks)
    }
  }, [mocks])

  return (
    <QueryClientProvider client={queryClient}>
      <Story />
    </QueryClientProvider>
  )
}
