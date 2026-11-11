import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  MemoryExperiment,
  OrganizationMemory,
  UpdateOrganizationMemoryRequest,
  UpdateWorkspaceMemoryRequest,
  WorkspaceMemory,
  WorkspaceMemoryErased,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  ORGANIZATION_MEMORY,
  WORKSPACES,
  workspaceMeta,
} from "@/shared/api/queryKeys"

function workspaceMemoryKey(workspaceId: string) {
  return [WORKSPACES, workspaceId, "memory"]
}

function memoryExperimentKey(workspaceId: string) {
  return [WORKSPACES, workspaceId, "memory-experiment"]
}

// The caller's organization's switch. Read only by its owners and admins, which
// is who the route answers, so the caller passes whether to ask at all.
export function useOrganizationMemory(enabled: boolean) {
  return useQuery({
    queryKey: [ORGANIZATION_MEMORY],
    queryFn: () => apiFetch<OrganizationMemory>("/organizations/me/memory"),
    enabled,
    staleTime: 60_000,
  })
}

export function useSetOrganizationMemory() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateOrganizationMemoryRequest) =>
      apiFetch<OrganizationMemory>("/organizations/me/memory", {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData([ORGANIZATION_MEMORY], data)
      // Every workspace's view says whether its organization has memory on.
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
    },
  })
}

export function useWorkspaceMemory(workspaceId: string) {
  return useQuery({
    queryKey: workspaceMemoryKey(workspaceId),
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceMemory>(
        `/workspaces/${encodeURIComponent(workspaceId)}/memory`,
      ),
    staleTime: 30_000,
  })
}

// The switches replace each other whole, so a change sends every one, as the page holds them.
export function useSetWorkspaceMemory() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: UpdateWorkspaceMemoryRequest
    }) =>
      apiFetch<WorkspaceMemory>(
        `/workspaces/${encodeURIComponent(workspaceId)}/memory`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: (data, { workspaceId }) => {
      queryClient.setQueryData(workspaceMemoryKey(workspaceId), data)
      // Starting, stopping or restarting the experiment changes what it reports.
      void queryClient.invalidateQueries({
        queryKey: memoryExperimentKey(workspaceId),
      })
    },
  })
}

// What memory saves the workspace: the running savings experiment, or the last one.
export function useMemoryExperiment(workspaceId: string, enabled: boolean) {
  return useQuery({
    queryKey: memoryExperimentKey(workspaceId),
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<MemoryExperiment>(
        `/workspaces/${encodeURIComponent(workspaceId)}/memory/experiment`,
      ),
    enabled,
    staleTime: 30_000,
  })
}

export function useEraseWorkspaceMemory() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ workspaceId }: { workspaceId: string }) =>
      apiFetch<WorkspaceMemoryErased>(
        `/workspaces/${encodeURIComponent(workspaceId)}/memory`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: workspaceMemoryKey(workspaceId),
      })
    },
  })
}
