import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  AgentRecommender,
  AgentRecommenders,
  CreateAgentRecommenderRequest,
  UpdateAgentRecommenderRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { AGENT_RECOMMENDERS } from "@/shared/api/queryKeys"

// The subagent routers: the decision models that pick a subagent's model when
// a coding agent asks (`/routing/recommenders`). Operator-only: the API answers
// a non-operator 404, which is why the page asks the deployment admin gate
// before it asks for this list.
//
// One router is on at a time, and switching one on switches the others off on
// the server, so every write refetches the whole list rather than patching the
// row it wrote.

const recommenderPath = (id: string) =>
  `/routing/recommenders/${encodeURIComponent(id)}`

function useInvalidateRecommenders() {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({ queryKey: [AGENT_RECOMMENDERS] })
  }
}

export function useAgentRecommenders(enabled = true) {
  return useQuery({
    queryKey: [AGENT_RECOMMENDERS, "list"],
    queryFn: () => apiFetch<AgentRecommenders>("/routing/recommenders"),
    staleTime: 30_000,
    enabled,
  })
}

export function useAddAgentRecommender() {
  const invalidate = useInvalidateRecommenders()
  return useMutation({
    mutationFn: (body: CreateAgentRecommenderRequest) =>
      apiFetch<AgentRecommender>("/routing/recommenders", {
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
      body,
    }: {
      id: string
      body: UpdateAgentRecommenderRequest
    }) =>
      apiFetch<AgentRecommender>(recommenderPath(id), {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: invalidate,
  })
}

export function useDeleteAgentRecommender() {
  const invalidate = useInvalidateRecommenders()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(recommenderPath(id), { method: "DELETE" }),
    onSuccess: invalidate,
  })
}
