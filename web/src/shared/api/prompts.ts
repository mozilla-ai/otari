import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CreatePromptRequest,
  StoredPrompt,
  UpdatePromptRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { PROMPTS } from "@/shared/api/queryKeys"

export function usePrompts() {
  return useQuery({
    queryKey: [PROMPTS],
    queryFn: () => apiFetch<StoredPrompt[]>("/prompts"),
    staleTime: 30_000,
  })
}

export function useCreatePrompt() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreatePromptRequest) =>
      apiFetch<StoredPrompt>("/prompts", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [PROMPTS] }),
  })
}

export function useUpdatePrompt() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdatePromptRequest }) =>
      apiFetch<StoredPrompt>(`/prompts/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [PROMPTS] }),
  })
}

export function useDeletePrompt() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/prompts/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [PROMPTS] }),
  })
}
