import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  AgentRecommender,
  AgentRecommenders,
  CreateAgentRecommenderRequest,
  UpdateAgentRecommenderRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { AGENT_RECOMMENDERS } from "@/shared/api/queryKeys"

// The decision models that answer `POST /routing/recommend`
// (`/routing/recommenders`). Operator-only: the API answers a non-operator
// 404, which is why the page asks the deployment admin gate before it asks
// for this list.

const PATH = "/routing/recommenders"

export function useAgentRecommenders(enabled = true) {
  return useQuery({
    queryKey: [AGENT_RECOMMENDERS],
    queryFn: () => apiFetch<AgentRecommenders>(PATH),
    staleTime: 30_000,
    enabled,
  })
}

/** Refetches the list after each write: switching one on switches the others off. */
function useInvalidateRecommenders() {
  const queryClient = useQueryClient()
  return () => queryClient.invalidateQueries({ queryKey: [AGENT_RECOMMENDERS] })
}

export function useAddAgentRecommender() {
  const invalidate = useInvalidateRecommenders()
  return useMutation({
    mutationFn: (body: CreateAgentRecommenderRequest) =>
      apiFetch<AgentRecommender>(PATH, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: invalidate,
  })
}

export function useUpdateAgentRecommender() {
  const invalidate = useInvalidateRecommenders()
  return useMutation({
    mutationFn: ({
      id,
      patch,
    }: {
      id: string
      patch: UpdateAgentRecommenderRequest
    }) =>
      apiFetch<AgentRecommender>(`${PATH}/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(patch),
      }),
    onSuccess: invalidate,
  })
}

export function useRemoveAgentRecommender() {
  const invalidate = useInvalidateRecommenders()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`${PATH}/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: invalidate,
  })
}
