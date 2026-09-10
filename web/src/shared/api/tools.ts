import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CreateOrganizationGuardrailRequest,
  CreateSearchToolRequest,
  CreateWorkspaceMcpServerRequest,
  OrganizationGuardrail,
  SearchProviderInfo,
  SearchToolsResponse,
  StoredSearchTool,
  TestServiceResponse,
  ToolSettingsResponse,
  ToolsResponse,
  UpdateOrganizationGuardrailRequest,
  UpdateSearchToolRequest,
  UpdateToolSettingsRequest,
  UpdateWorkspaceCodeExecutionPolicyRequest,
  UpdateWorkspaceMcpServerRequest,
  UpdateWorkspaceWebSearchConfigRequest,
  WorkspaceCodeExecutionPolicy,
  WorkspaceMcpServer,
  WorkspaceMcpServers,
  WorkspaceWebSearchConfig,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
import {
  ORGANIZATION_GUARDRAILS,
  SEARCH_PROVIDERS,
  SEARCH_TOOLS,
  TOOL_SETTINGS,
  TOOLS,
  WORKSPACES,
} from "@/shared/api/queryKeys"

export function useToolSettings(enabled = true) {
  return useQuery({
    queryKey: [TOOL_SETTINGS],
    queryFn: () => apiFetch<ToolSettingsResponse>("/tool-settings"),
    staleTime: 60_000,
    enabled,
  })
}

// The declaration forms this deployment honors. Depends on tool settings
// (interception, the backend URLs), so a settings save invalidates it.
export function useTools(enabled = true) {
  return useQuery({
    queryKey: [TOOLS],
    queryFn: () => apiFetch<ToolsResponse>("/tools"),
    staleTime: 60_000,
    enabled,
  })
}

export function useUpdateToolSettings() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateToolSettingsRequest) =>
      apiFetch<ToolSettingsResponse>("/tool-settings", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData([TOOL_SETTINGS], data)
      // Toggling interception or clearing a backend URL changes which
      // declarations the gateway accepts, so the "how to call" card must refetch.
      void queryClient.invalidateQueries({ queryKey: [TOOLS] })
      // A searxng search tool with no api_base of its own inherits web_search_url,
      // which this PATCH may have just changed, so the endpoint a blank box
      // resolves to (and whether one is required at all) has to be re-read.
      void queryClient.invalidateQueries({ queryKey: [SEARCH_PROVIDERS] })
    },
  })
}

// Search tools served by POST /v1/search: the editable rows plus the read-only
// config-file entries, so the page can show every tool a caller could name.
export function useSearchTools() {
  return useQuery({
    queryKey: [SEARCH_TOOLS],
    queryFn: () => apiFetch<SearchToolsResponse>("/search-tools"),
    staleTime: 60_000,
  })
}

// Which search providers this build supports, and what each one needs, so the
// add form asks for a key or a backend URL only when the provider requires it.
export function useSearchProviders() {
  return useQuery({
    queryKey: [SEARCH_PROVIDERS],
    queryFn: () => apiFetch<SearchProviderInfo[]>("/search-tools/providers"),
    staleTime: 300_000,
  })
}

export function useCreateSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateSearchToolRequest) =>
      apiFetch<StoredSearchTool>("/search-tools", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] }),
  })
}

export function useUpdateSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      name,
      body,
    }: {
      name: string
      body: UpdateSearchToolRequest
    }) =>
      apiFetch<StoredSearchTool>(`/search-tools/${encodeURIComponent(name)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] }),
  })
}

export function useDeleteSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<void>(`/search-tools/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] }),
  })
}

// Probe a (typically unsaved) service URL for reachability. Read-only, so it
// invalidates nothing.
export function useTestService() {
  return useMutation({
    mutationFn: ({ service, url }: { service: string; url: string }) =>
      apiFetch<TestServiceResponse>(
        `/tool-settings/${encodeURIComponent(service)}/test`,
        {
          method: "POST",
          body: JSON.stringify({ url }),
        },
      ),
  })
}

// The pricing endpoint caps `limit` at 1000 server-side, so page through it
// rather than truncating: a gateway with a long price history could otherwise
// have older rows silently vanish from the models table.

export function useOrganizationGuardrails(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_GUARDRAILS],
    queryFn: () =>
      fetchAllPaged<OrganizationGuardrail>("/organizations/me/guardrails"),
    staleTime: 60_000,
    enabled,
  })
}

export function useCreateOrganizationGuardrail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateOrganizationGuardrailRequest) =>
      apiFetch<OrganizationGuardrail>("/organizations/me/guardrails", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_GUARDRAILS],
      })
    },
  })
}

export function useUpdateOrganizationGuardrail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      guardrailId,
      body,
    }: {
      guardrailId: string
      body: UpdateOrganizationGuardrailRequest
    }) =>
      apiFetch<OrganizationGuardrail>(
        `/organizations/me/guardrails/${encodeURIComponent(guardrailId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_GUARDRAILS],
      })
    },
  })
}

export function useDeleteOrganizationGuardrail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (guardrailId: string) =>
      apiFetch<{ message: string }>(
        `/organizations/me/guardrails/${encodeURIComponent(guardrailId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [ORGANIZATION_GUARDRAILS],
      })
    },
  })
}

export function useWorkspaceCodeExecutionPolicy(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "code-execution-policy"],
    queryFn: () =>
      apiFetch<WorkspaceCodeExecutionPolicy>(
        `/workspaces/${encodeURIComponent(workspaceId as string)}/code-execution-policy`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useSetWorkspaceCodeExecutionPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: UpdateWorkspaceCodeExecutionPolicyRequest
    }) =>
      apiFetch<WorkspaceCodeExecutionPolicy>(
        `/workspaces/${encodeURIComponent(workspaceId)}/code-execution-policy`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: (data, { workspaceId }) => {
      // Same as the web-search write above: the response is the stored row.
      queryClient.setQueryData(
        [WORKSPACES, workspaceId, "code-execution-policy"],
        data,
      )
    },
  })
}

// Drops the row, which returns the workspace to the deployment's own behavior.
// Not the same as saving `enabled: true`: that is a stored decision not to
// narrow, while this is no decision at all.
export function useClearWorkspaceCodeExecutionPolicy() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ workspaceId }: { workspaceId: string }) =>
      apiFetch<WorkspaceCodeExecutionPolicy>(
        `/workspaces/${encodeURIComponent(workspaceId)}/code-execution-policy`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "code-execution-policy"],
      })
    },
  })
}

// A workspace's web-search configuration over the deployment-wide backend. One
// object or none, so it is a plain read rather than a paged list, and it is
// nested under the workspaces key for the same reason the code-execution policy
// next door is.
export function useWorkspaceWebSearchConfig(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "web-search"],
    queryFn: () =>
      apiFetch<WorkspaceWebSearchConfig>(
        `/workspaces/${encodeURIComponent(workspaceId as string)}/web-search`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useSetWorkspaceWebSearchConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: UpdateWorkspaceWebSearchConfigRequest
    }) =>
      apiFetch<WorkspaceWebSearchConfig>(
        `/workspaces/${encodeURIComponent(workspaceId)}/web-search`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: (data, { workspaceId }) => {
      // The PUT answers with the row it just stored, so seeding the cache with
      // it is both fresher and cheaper than refetching: without this the query
      // holds the pre-write row until a GET lands, and each save costs two
      // requests instead of one.
      queryClient.setQueryData([WORKSPACES, workspaceId, "web-search"], data)
    },
  })
}

// Drops the row, which returns the workspace to the deployment's own behavior.
// Not the same as saving `enabled: true`: that is a stored decision not to
// narrow, while this is no decision at all.
export function useClearWorkspaceWebSearchConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ workspaceId }: { workspaceId: string }) =>
      apiFetch<WorkspaceWebSearchConfig>(
        `/workspaces/${encodeURIComponent(workspaceId)}/web-search`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "web-search"],
      })
    },
  })
}

// A workspace's MCP servers. A list rather than the single row the two config
// planes beside it hold, and nested under the workspaces key for the same
// reason they are.
//
// One request, at the endpoint's own documented ceiling, rather than a paged
// walk: the service caps how many servers a workspace may hold well below this,
// so a second page cannot exist. The ceiling is the route's (`le=1000`), not a
// copy of that cap, so this stays right if the cap moves.
const MCP_SERVERS_PAGE_SIZE = 1000

export function useWorkspaceMcpServers(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
    queryFn: () =>
      apiFetch<WorkspaceMcpServers>(
        `/workspaces/${encodeURIComponent(workspaceId as string)}/mcp-servers?limit=${MCP_SERVERS_PAGE_SIZE}`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useCreateWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: CreateWorkspaceMcpServerRequest
    }) =>
      apiFetch<WorkspaceMcpServer>(
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}

// A partial update, which is what keeps the write-only token's three states
// expressible. See `McpServerDialog` for the rule and how the form maps onto it.
export function useUpdateWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      body,
    }: {
      workspaceId: string
      serverId: string
      body: UpdateWorkspaceMcpServerRequest
    }) =>
      apiFetch<WorkspaceMcpServer>(
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/${encodeURIComponent(serverId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}

export function useDeleteWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
    }: {
      workspaceId: string
      serverId: string
    }) =>
      apiFetch<void>(
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/${encodeURIComponent(serverId)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}
