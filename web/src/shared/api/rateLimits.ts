import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CreateRateLimitRuleRequest,
  RateLimitRule,
  RateLimitRules,
  UpdateRateLimitRuleRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { RATE_LIMIT_RULES } from "@/shared/api/queryKeys"

// Every rule in effect: the config.yml ones, read-only, then the stored ones.
export function useRateLimitRules() {
  return useQuery({
    queryKey: [RATE_LIMIT_RULES],
    queryFn: () => apiFetch<RateLimitRules>("/rate-limits"),
    staleTime: 60_000,
  })
}

export function useCreateRateLimitRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateRateLimitRuleRequest) =>
      apiFetch<RateLimitRule>("/rate-limits", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [RATE_LIMIT_RULES] }),
  })
}

export function useUpdateRateLimitRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      name,
      body,
    }: {
      name: string
      body: UpdateRateLimitRuleRequest
    }) =>
      apiFetch<RateLimitRule>(`/rate-limits/${encodeURIComponent(name)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [RATE_LIMIT_RULES] }),
  })
}

export function useDeleteRateLimitRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<void>(`/rate-limits/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [RATE_LIMIT_RULES] }),
  })
}
