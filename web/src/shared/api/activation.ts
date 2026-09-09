import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type { ActivationApiKey, WorkspaceActivation } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { ACTIVATION, KEYS, NO_RETRY } from "@/shared/api/queryKeys"

// ---------------------------------------------------------------------------
// The first-request setup guide
// ---------------------------------------------------------------------------

// How often the guide asks whether the workspace's first request has landed.
// Only while it is on screen and still waiting (the card passes `enabled`), and
// the answer is one or two indexed reads server-side, so this is the interval
// that makes "send the request, watch it arrive" feel live without polling for a
// dashboard nobody is looking at.
const ACTIVATION_POLL_MS = 4_000

/**
 * Where the selected workspace stands on its first successful request.
 *
 * Polls only while the guide is actually being offered, which is also the only
 * state whose answer can still change on its own. A workspace that activated
 * cannot go back, and one whose guide was dismissed (or turned off for the
 * deployment) has nothing to wait for, so both stop the interval rather than
 * asking every few seconds for the life of the page.
 */
export function useWorkspaceActivation(
  workspaceId: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: [ACTIVATION, workspaceId],
    queryFn: () =>
      apiFetch<WorkspaceActivation>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/activation`,
      ),
    enabled: enabled && workspaceId !== null,
    refetchInterval: (query) =>
      query.state.data?.experience_eligible ? ACTIVATION_POLL_MS : false,
    // A failed check is reported on the card, which offers "Check now": retrying
    // twice behind the operator's back would only delay that by a poll interval.
    ...NO_RETRY,
  })
}

// Issues the workspace's setup key and returns its plaintext exactly once, like
// `useCreateKey`: the caller shows it and must never write the response into the
// query cache. KEYS is invalidated because the key it rotates is an ordinary row
// on the Keys page, and ACTIVATION because the guide's state now records it.
export function useCreateActivationKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (workspaceId: string) =>
      apiFetch<ActivationApiKey>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/activation/key`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [KEYS] })
      void queryClient.invalidateQueries({ queryKey: [ACTIVATION] })
    },
  })
}

// Permanent, and idempotent server-side. Only ACTIVATION is invalidated:
// dismissing retires the card and leaves the key it issued alone, so nothing on
// the Keys page changed.
export function useDismissActivation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (workspaceId: string) =>
      apiFetch<{ message: string }>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/activation/dismiss`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ACTIVATION] })
    },
  })
}
