import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type { UpdateCheckAnswer, UpdateCheckStatus } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { UPDATE_CHECK } from "@/shared/api/queryKeys"

/**
 * Whether the deployment checks for new releases, and what the last check
 * heard. Operator-only on the server, so `enabled` carries the caller's
 * standing and the surface: a tenant never asks, and neither does a deployment
 * that does not publish `update_check`. A check lands at most once a day, so a
 * long `staleTime` loses nothing.
 */
export function useUpdateCheck(enabled: boolean) {
  return useQuery({
    queryKey: [UPDATE_CHECK],
    queryFn: () => apiFetch<UpdateCheckStatus>("/update-check"),
    enabled,
    staleTime: 5 * 60_000,
  })
}

/** Record an operator's answer. The status comes back with it. */
export function useAnswerUpdateCheck() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateCheckAnswer) =>
      apiFetch<UpdateCheckStatus>("/update-check", {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: (status) => queryClient.setQueryData([UPDATE_CHECK], status),
  })
}
