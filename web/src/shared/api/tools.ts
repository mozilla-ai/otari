import {
  keepPreviousData,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  CheckWorkspaceMcpServerRequest,
  CreatedSearchTool,
  CreateOrganizationGuardrailRequest,
  CreateSearchToolRequest,
  CreateWorkspaceMcpServerGrantsRequest,
  CreateWorkspaceMcpServerRequest,
  GuardrailCatalog,
  McpConnectionCompleted,
  McpConnectionStarted,
  McpConnections,
  McpServerConnections,
  OrganizationGuardrail,
  OrganizationGuardrailTestResult,
  SandboxCredentialPublic,
  SandboxCredentialRequest,
  SearchProviderInfo,
  SearchToolsResponse,
  SearchToolTestRequest,
  SearchToolTestResponse,
  StoredSearchTool,
  StoredSearchToolTestRequest,
  TestOrganizationGuardrailRequest,
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
  WorkspaceMcpServerAccess,
  WorkspaceMcpServerAccessCheck,
  WorkspaceMcpServerHealth,
  WorkspaceMcpServerKeyAccess,
  WorkspaceMcpServerLogo,
  WorkspaceMcpServerLogos,
  WorkspaceMcpServerMyTools,
  WorkspaceMcpServers,
  WorkspaceWebSearchConfig,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
import {
  GUARDRAIL_PROFILES,
  MCP_SERVER_REQUESTS,
  ORGANIZATION_GUARDRAILS,
  PLAYGROUND,
  SEARCH_PROVIDERS,
  SEARCH_TOOLS,
  TOOL_SETTINGS,
  TOOLS,
  WORKSPACES,
  workspaceMeta,
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
      // Same reasoning one service over: the guardrail catalog is whatever the
      // host `guardrails_url` names answered with, so pointing that field at a
      // different sidecar changes which profiles exist.
      void queryClient.invalidateQueries({ queryKey: [GUARDRAIL_PROFILES] })
    },
  })
}

// The key a hosted sandbox provider leases with. Never read back: the answer is
// where it comes from and its last four, which the settings query carries.
function useSandboxCredentialWrite<TVariables>(
  request: (variables: TVariables) => Promise<SandboxCredentialPublic>,
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: request,
    onSuccess: (credential) => {
      queryClient.setQueryData<ToolSettingsResponse>([TOOL_SETTINGS], (old) =>
        old ? { ...old, sandbox_credential: credential } : old,
      )
    },
  })
}

export function useSetSandboxCredential() {
  return useSandboxCredentialWrite((body: SandboxCredentialRequest) =>
    apiFetch<SandboxCredentialPublic>("/tool-settings/sandbox/credential", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  )
}

export function useClearSandboxCredential() {
  return useSandboxCredentialWrite(() =>
    apiFetch<SandboxCredentialPublic>("/tool-settings/sandbox/credential", {
      method: "DELETE",
    }),
  )
}

// Search tools served by POST /v1/search: the editable rows plus the read-only
// config-file entries, so the page can show every tool a caller could name.
// `fetch` lists the fetch instances instead, `builtin_fetch` among them.
export function useSearchTools(
  kind: "search" | "fetch" = "search",
  enabled = true,
) {
  return useQuery({
    queryKey: [SEARCH_TOOLS, kind],
    queryFn: () =>
      apiFetch<SearchToolsResponse>(
        kind === "search" ? "/search-tools" : "/search-tools?kind=fetch",
      ),
    staleTime: 60_000,
    enabled,
  })
}

// Which search or fetch providers this build supports, what each one needs and
// the options it takes, so the add form asks only for what the provider
// requires. Keyed by kind under `SEARCH_PROVIDERS`, so `useUpdateToolSettings`'s
// invalidation reaches both: a `searxng` tool's inherited endpoint follows
// `web_search_url`.
export function useSearchProviders(
  kind: "search" | "fetch" = "search",
  enabled = true,
) {
  return useQuery({
    queryKey: [SEARCH_PROVIDERS, kind],
    queryFn: () =>
      apiFetch<SearchProviderInfo[]>(
        kind === "search"
          ? "/search-tools/providers"
          : "/search-tools/providers?kind=fetch",
      ),
    staleTime: 300_000,
    enabled,
  })
}

// A create can store `web_search_default_tool` (the pin its `notice` reports)
// and can turn the model's search on, so the settings and the tool list are
// read again with the tools; the catalog lists each provider's tools by name.
export function useCreateSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateSearchToolRequest) =>
      apiFetch<CreatedSearchTool>("/search-tools", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] })
      void queryClient.invalidateQueries({ queryKey: [SEARCH_PROVIDERS] })
      void queryClient.invalidateQueries({ queryKey: [TOOL_SETTINGS] })
      void queryClient.invalidateQueries({ queryKey: [TOOLS] })
    },
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
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] })
      void queryClient.invalidateQueries({ queryKey: [SEARCH_PROVIDERS] })
    },
  })
}

// A default that names the deleted tool reads as unset, so the settings are
// read again with the tools.
export function useDeleteSearchTool() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<void>(`/search-tools/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [SEARCH_TOOLS] })
      void queryClient.invalidateQueries({ queryKey: [SEARCH_PROVIDERS] })
      void queryClient.invalidateQueries({ queryKey: [TOOL_SETTINGS] })
      void queryClient.invalidateQueries({ queryKey: [TOOLS] })
    },
  })
}

// One search or one fetch through a configuration that is not saved yet.
// Read-only, so it invalidates nothing.
export function useTestUnsavedSearchTool() {
  return useMutation({
    mutationFn: (body: SearchToolTestRequest) =>
      apiFetch<SearchToolTestResponse>("/search-tools/test", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// One search or one fetch through a configured or stored tool, `builtin_fetch`
// included. Read-only, so it invalidates nothing.
export function useTestSearchTool() {
  return useMutation({
    mutationFn: ({
      name,
      body,
    }: {
      name: string
      body: StoredSearchToolTestRequest
    }) =>
      apiFetch<SearchToolTestResponse>(
        `/search-tools/${encodeURIComponent(name)}/test`,
        { method: "POST", body: JSON.stringify(body) },
      ),
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

// The profiles an organization guardrail may name, and the validate_kwargs each
// one takes. Read from the guardrails service through the gateway, so an
// unreachable or unconfigured service resolves to `available: false` with a
// reason rather than to a query error: the form falls back to naming a profile
// by hand and has to render either way.
//
// Longer-lived than the tool settings beside it, because the answer only changes
// when the operator edits the sidecar's own YAML and restarts it, which is not
// something the dashboard can do. The window `useSearchProviders` takes, for the
// reason it takes it.
export function useGuardrailProfiles(enabled = true) {
  return useQuery({
    queryKey: [GUARDRAIL_PROFILES],
    queryFn: () =>
      apiFetch<GuardrailCatalog>("/tool-settings/guardrails/profiles"),
    staleTime: 300_000,
    enabled,
  })
}

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

export function useTestOrganizationGuardrail() {
  return useMutation({
    mutationFn: ({
      guardrailId,
      body,
    }: {
      guardrailId: string
      body: TestOrganizationGuardrailRequest
    }) =>
      apiFetch<OrganizationGuardrailTestResult>(
        `/organizations/me/guardrails/${encodeURIComponent(guardrailId)}/test`,
        { method: "POST", body: JSON.stringify(body) },
      ),
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
    meta: workspaceMeta(workspaceId),
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
    meta: workspaceMeta(workspaceId),
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
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceMcpServers>(
        `/workspaces/${encodeURIComponent(workspaceId as string)}/mcp-servers?limit=${MCP_SERVERS_PAGE_SIZE}`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

// One stored server, for its own page. Under the list's key, so every write that
// refreshes the list refreshes this too.
export function useWorkspaceMcpServer(
  workspaceId: string | null,
  serverId: string,
) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "mcp-servers", serverId],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceMcpServer>(
        mcpServerPath(workspaceId as string, serverId),
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

/**
 * The logo of each of a workspace's servers, by server id: what each one shows
 * and whether it needs a tile. One page at the servers list's own size, so it
 * covers every server that list shows. Under the servers' key, so a write that refreshes
 * the list (a new server, a changed URL) refreshes this too. Reading it makes
 * the gateway fetch nothing; an image URL without a version does, once.
 */
export function useWorkspaceMcpServerLogos(workspaceId: string | null) {
  const query = useQuery({
    queryKey: [WORKSPACES, workspaceId, "mcp-servers", "logos"],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceMcpServerLogos>(
        `/workspaces/${encodeURIComponent(workspaceId as string)}/mcp-servers/logos?limit=${MCP_SERVERS_PAGE_SIZE}`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
  return new Map((query.data?.data ?? []).map((logo) => [logo.server_id, logo]))
}

/** An admin's own logo for a server and the tile choice, saved together. */
export function useSetWorkspaceMcpServerLogo() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      tile,
      png,
    }: {
      workspaceId: string
      serverId: string
      tile: WorkspaceMcpServerLogo["logo_tile"]
      /** Absent to keep the logo it has. */
      png?: Blob
    }) => {
      const body = new FormData()
      body.set("tile", tile)
      if (png) body.set("file", png, "logo.png")
      return apiFetch<WorkspaceMcpServerLogo>(
        `${mcpServerPath(workspaceId, serverId)}/logo`,
        { method: "PUT", body },
      )
    },
    onSuccess: (_logo, { workspaceId }) =>
      queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers", "logos"],
      }),
  })
}

/** Remove an admin's uploaded logo, so the server shows its own again. */
export function useRemoveWorkspaceMcpServerLogo() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
    }: {
      workspaceId: string
      serverId: string
    }) =>
      apiFetch<WorkspaceMcpServerLogo>(
        `${mcpServerPath(workspaceId, serverId)}/logo`,
        { method: "DELETE" },
      ),
    onSuccess: (_logo, { workspaceId }) =>
      queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers", "logos"],
      }),
  })
}

function mcpServerPath(workspaceId: string, serverId: string): string {
  return `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/${encodeURIComponent(serverId)}`
}

function mcpAccessKey(workspaceId: string, serverId: string) {
  return [WORKSPACES, workspaceId, "mcp-servers", serverId, "access"] as const
}

// What each key reaches across the workspace's servers. A segment of its own
// beside the server ids, so a grant write can refresh every key's view without
// refetching the servers.
function mcpKeyAccessRoot(workspaceId: string) {
  return [WORKSPACES, workspaceId, "mcp-servers", "keys"] as const
}

// A server's access list: `everyone_tools` and the grants that override it per
// key or per person. Manager-only, so a viewer's 403 is the answer, not a fault.
export function useWorkspaceMcpServerAccess(
  workspaceId: string | null,
  serverId: string,
  enabled = true,
) {
  return useQuery({
    queryKey: mcpAccessKey(workspaceId ?? "", serverId),
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceMcpServerAccess>(
        `${mcpServerPath(workspaceId as string, serverId)}/access`,
      ),
    enabled: enabled && workspaceId !== null,
    staleTime: 30_000,
  })
}

// Whether one key may call one tool, and which rule decides. Read on demand by
// the page's test, so it is keyed on what was asked and never kept fresh.
export function useWorkspaceMcpServerAccessCheck({
  workspaceId,
  serverId,
  apiKeyId,
  tool,
}: {
  workspaceId: string
  serverId: string
  apiKeyId: string
  tool: string
}) {
  return useQuery({
    queryKey: [...mcpAccessKey(workspaceId, serverId), "check", apiKeyId, tool],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceMcpServerAccessCheck>(
        `${mcpServerPath(workspaceId, serverId)}/access/check?api_key_id=${encodeURIComponent(apiKeyId)}&tool=${encodeURIComponent(tool)}`,
      ),
    enabled: apiKeyId !== "" && tool !== "",
    staleTime: 0,
  })
}

// What one key reaches on each of the workspace's servers, for the key editor.
export function useWorkspaceMcpKeyAccess(
  workspaceId: string,
  apiKeyId: string,
  enabled = true,
) {
  return useQuery({
    queryKey: [...mcpKeyAccessRoot(workspaceId), apiKeyId],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<WorkspaceMcpServerKeyAccess[]>(
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/keys/${encodeURIComponent(apiKeyId)}/access`,
      ),
    enabled,
    staleTime: 30_000,
  })
}

// Every grant write answers with the whole access list, so the cache is seeded
// from it rather than refetched. A grant moves what a key reaches and what a
// check answers, so those two are refreshed.
function useGrantWrite<
  Variables extends { workspaceId: string; serverId: string },
>(request: (variables: Variables) => Promise<WorkspaceMcpServerAccess>) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: request,
    onSuccess: (access, { workspaceId, serverId }) => {
      queryClient.setQueryData(mcpAccessKey(workspaceId, serverId), access)
      void queryClient.invalidateQueries({
        queryKey: [...mcpAccessKey(workspaceId, serverId), "check"],
      })
      void queryClient.invalidateQueries({
        queryKey: mcpKeyAccessRoot(workspaceId),
      })
      void queryClient.invalidateQueries({
        queryKey: mcpMyToolsKey(workspaceId, serverId),
      })
    },
  })
}

export function useAddWorkspaceMcpServerGrants() {
  return useGrantWrite(
    ({
      workspaceId,
      serverId,
      body,
    }: {
      workspaceId: string
      serverId: string
      body: CreateWorkspaceMcpServerGrantsRequest
    }) =>
      apiFetch<WorkspaceMcpServerAccess>(
        `${mcpServerPath(workspaceId, serverId)}/access/grants`,
        { method: "POST", body: JSON.stringify(body) },
      ),
  )
}

export function useUpdateWorkspaceMcpServerGrant() {
  return useGrantWrite(
    ({
      workspaceId,
      serverId,
      grantId,
      tools,
    }: {
      workspaceId: string
      serverId: string
      grantId: string
      tools: string[]
    }) =>
      apiFetch<WorkspaceMcpServerAccess>(
        `${mcpServerPath(workspaceId, serverId)}/access/grants/${encodeURIComponent(grantId)}`,
        { method: "PATCH", body: JSON.stringify({ tools }) },
      ),
  )
}

export function useDeleteWorkspaceMcpServerGrant() {
  return useGrantWrite(
    ({
      workspaceId,
      serverId,
      grantId,
    }: {
      workspaceId: string
      serverId: string
      grantId: string
    }) =>
      apiFetch<WorkspaceMcpServerAccess>(
        `${mcpServerPath(workspaceId, serverId)}/access/grants/${encodeURIComponent(grantId)}`,
        { method: "DELETE" },
      ),
  )
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
      // Storing a server approves the members' requests for it.
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, MCP_SERVER_REQUESTS],
      })
    },
  })
}

// A partial update, which is what keeps the write-only token's three states
// expressible. See `McpServerSettings` for the rule and how the form maps onto it.
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
      apiFetch<WorkspaceMcpServer>(mcpServerPath(workspaceId, serverId), {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: (server, { workspaceId, serverId }) => {
      queryClient.setQueryData(
        [WORKSPACES, workspaceId, "mcp-servers", serverId],
        server,
      )
      // The prefix covers the server's access list and every key's view of it,
      // which `everyone_tools` and `enabled` both move.
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}

// Connects to a stored server now and records the outcome as its `health`. A
// server that fails answers 200 like one that passes; only a busy gateway (429)
// is an error here.
export function useCheckWorkspaceMcpServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
    }: {
      workspaceId: string
      serverId: string
    }) =>
      apiFetch<WorkspaceMcpServer>(
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/${encodeURIComponent(serverId)}/check`,
        { method: "POST" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
    },
  })
}

// Checks connection details without storing anything: the dialog's Test
// button. Nothing to invalidate, since nothing was written.
export function useTestWorkspaceMcpServerConnection() {
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: CheckWorkspaceMcpServerRequest
    }) =>
      apiFetch<WorkspaceMcpServerHealth>(
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/check`,
        { method: "POST", body: JSON.stringify(body) },
      ),
  })
}

// A probe answers for the server as it was a moment ago, which is all a
// suggested default needs; re-asking while the review step is open would only
// slow the add down.
const MCP_SERVER_PROBE_STALE_MS = 5 * 60_000

/**
 * A connection check of each of `urls` with no credential, without storing
 * anything: what a server says to a caller who has not signed in. The answer
 * suggests a connection mode, so nothing waits on it and a refusal of the
 * check itself (a busy gateway) is simply no suggestion.
 */
export function useMcpServerProbes(workspaceId: string, urls: string[]) {
  return useQueries({
    queries: urls.map((url) => ({
      queryKey: [WORKSPACES, workspaceId, "mcp-server-probe", url],
      staleTime: MCP_SERVER_PROBE_STALE_MS,
      retry: false,
      queryFn: () =>
        apiFetch<WorkspaceMcpServerHealth>(
          `/workspaces/${encodeURIComponent(workspaceId)}/mcp-servers/check`,
          {
            method: "POST",
            body: JSON.stringify({ url, authorization_token: "" }),
          },
        ),
    })),
  })
}

function mcpConnectionsKey(workspaceId: string) {
  return [WORKSPACES, workspaceId, "mcp-connections"] as const
}

/** The Playground's tools menu says which servers still need a sign-in. */
function playgroundToolsKey(workspaceId: string) {
  return [PLAYGROUND, workspaceId, "tools"] as const
}

// Under the server's key, so a check, an edit, a finished sign-in or a
// connection made through `useCompleteMcpConnection` refreshes it with the server.
function mcpMyToolsKey(workspaceId: string, serverId: string) {
  return [WORKSPACES, workspaceId, "mcp-servers", serverId, "my-tools"] as const
}

function mcpServerConnectionsKey(workspaceId: string, serverId: string) {
  return [
    WORKSPACES,
    workspaceId,
    "mcp-servers",
    serverId,
    "connections",
  ] as const
}

/** The caller's own sign-ins to this workspace's per-user servers. */
export function useMyMcpConnections(workspaceId: string | null) {
  return useQuery({
    queryKey: mcpConnectionsKey(workspaceId ?? ""),
    enabled: workspaceId !== null,
    // Every write that moves it invalidates it, so a minute between reads is safe.
    staleTime: 60_000,
    queryFn: ({ signal }) =>
      apiFetch<McpConnections>(
        `/workspaces/${encodeURIComponent(workspaceId as string)}/mcp-connections`,
        { signal },
      ),
  })
}

/**
 * The tools the caller may use on a server under their own rule. A check, an
 * edit of the server, a rule change, a connection and a disconnection each
 * refresh it.
 */
export function useMyMcpServerTools(
  workspaceId: string | null,
  serverId: string,
  enabled: boolean,
) {
  return useQuery({
    queryKey: mcpMyToolsKey(workspaceId ?? "", serverId),
    enabled: enabled && workspaceId !== null,
    staleTime: 60_000,
    queryFn: ({ signal }) =>
      apiFetch<WorkspaceMcpServerMyTools>(
        `${mcpServerPath(workspaceId as string, serverId)}/my-tools`,
        { signal },
      ),
  })
}

/**
 * Starts a manager's sign-in made only to check a Personal server: the answer
 * is where to send the browser. Finishing it records the check and connects
 * nobody.
 */
export function useStartMcpCheckSignIn() {
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      returnTo,
    }: {
      workspaceId: string
      serverId: string
      returnTo: string
    }) =>
      apiFetch<McpConnectionStarted>(
        `${mcpServerPath(workspaceId, serverId)}/check/sign-in`,
        { method: "POST", body: JSON.stringify({ return_to: returnTo }) },
      ),
  })
}

/**
 * Starts a manager's sign-in for everyone on a Shared server: the answer is
 * where to send the browser. Finishing it stores the account as the server's
 * shared account, in place of a token.
 */
export function useStartMcpSharedSignIn() {
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      returnTo,
    }: {
      workspaceId: string
      serverId: string
      returnTo: string
    }) =>
      apiFetch<McpConnectionStarted>(
        `${mcpServerPath(workspaceId, serverId)}/shared-account`,
        { method: "POST", body: JSON.stringify({ return_to: returnTo }) },
      ),
  })
}

/** Ends a Shared server's account. The answer is the server, checked again without it. */
export function useEndMcpSharedAccount() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
    }: {
      workspaceId: string
      serverId: string
    }) =>
      apiFetch<WorkspaceMcpServer>(
        `${mcpServerPath(workspaceId, serverId)}/shared-account`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
      })
      void queryClient.invalidateQueries({
        queryKey: playgroundToolsKey(workspaceId),
      })
    },
  })
}

/**
 * Refreshes what a finished sign-in in another window changed: the caller's
 * connections, the Playground's tools menu, the server's connected users, and,
 * after a check, the server itself. The window that finished it has a query
 * cache of its own, so this page's is refreshed here.
 */
export function useRefreshAfterMcpSignIn() {
  const queryClient = useQueryClient()
  return (workspaceId: string, serverId: string) => {
    for (const queryKey of [
      mcpConnectionsKey(workspaceId),
      playgroundToolsKey(workspaceId),
      [WORKSPACES, workspaceId, "mcp-servers"],
      mcpServerConnectionsKey(workspaceId, serverId),
    ]) {
      void queryClient.invalidateQueries({ queryKey })
    }
  }
}

/** Starts the caller's sign-in: the answer is where to send the browser. */
export function useStartMcpConnection() {
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      returnTo,
    }: {
      workspaceId: string
      serverId: string
      returnTo: string
    }) =>
      apiFetch<McpConnectionStarted>(
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(serverId)}`,
        { method: "POST", body: JSON.stringify({ return_to: returnTo }) },
      ),
  })
}

/** Finishes a sign-in from the code its authorization server returned. */
export function useCompleteMcpConnection() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      state,
      code,
      iss,
    }: {
      state: string
      code: string
      iss?: string
    }) =>
      apiFetch<McpConnectionCompleted>("/mcp-connections/complete", {
        method: "POST",
        // `JSON.stringify` drops an absent `iss`.
        body: JSON.stringify({ state, code, iss }),
      }),
    onSuccess: (done) => {
      // Every finished sign-in lists the server's tools once, so its health moved.
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, done.workspace_id, "mcp-servers"],
      })
      void queryClient.invalidateQueries({
        queryKey: mcpConnectionsKey(done.workspace_id),
      })
      void queryClient.invalidateQueries({
        queryKey: playgroundToolsKey(done.workspace_id),
      })
      void queryClient.invalidateQueries({
        queryKey: mcpServerConnectionsKey(done.workspace_id, done.server_id),
      })
    },
  })
}

/** Ends the caller's own connection to a server. */
export function useDisconnectMcp() {
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
        `/workspaces/${encodeURIComponent(workspaceId)}/mcp-connections/${encodeURIComponent(serverId)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId, serverId }) => {
      void queryClient.invalidateQueries({
        queryKey: mcpConnectionsKey(workspaceId),
      })
      void queryClient.invalidateQueries({
        queryKey: playgroundToolsKey(workspaceId),
      })
      void queryClient.invalidateQueries({
        queryKey: mcpServerConnectionsKey(workspaceId, serverId),
      })
      void queryClient.invalidateQueries({
        queryKey: mcpMyToolsKey(workspaceId, serverId),
      })
    },
  })
}

/** One page of who is connected to a per-user server. Workspace managers only. */
export function useMcpServerConnections(
  workspaceId: string,
  serverId: string,
  { page, pageSize }: { page: number; pageSize: number },
  enabled: boolean,
) {
  return useQuery({
    queryKey: [
      ...mcpServerConnectionsKey(workspaceId, serverId),
      page,
      pageSize,
    ],
    enabled,
    // Other people connect without this page hearing of it, so it is read again sooner.
    staleTime: 30_000,
    placeholderData: keepPreviousData,
    queryFn: ({ signal }) => {
      const params = new URLSearchParams({
        skip: String(page * pageSize),
        limit: String(pageSize),
      })
      return apiFetch<McpServerConnections>(
        `${mcpServerPath(workspaceId, serverId)}/connections?${params}`,
        { signal },
      )
    },
  })
}

/** Ends one user's connection to a server. Workspace managers only. */
export function useRemoveMcpServerConnection() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      serverId,
      userId,
    }: {
      workspaceId: string
      serverId: string
      userId: string
    }) =>
      apiFetch<void>(
        `${mcpServerPath(workspaceId, serverId)}/connections/${encodeURIComponent(userId)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId, serverId }) => {
      void queryClient.invalidateQueries({
        queryKey: mcpServerConnectionsKey(workspaceId, serverId),
      })
      void queryClient.invalidateQueries({
        queryKey: mcpConnectionsKey(workspaceId),
      })
      void queryClient.invalidateQueries({
        queryKey: playgroundToolsKey(workspaceId),
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
      apiFetch<void>(mcpServerPath(workspaceId, serverId), {
        method: "DELETE",
      }),
    onSuccess: (_data, { workspaceId, serverId }) => {
      // Dropped rather than refetched: the server's own page is still mounted
      // while it navigates away, and a refetch would only earn it a 404.
      queryClient.removeQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers", serverId],
      })
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "mcp-servers"],
        exact: true,
      })
      void queryClient.invalidateQueries({
        queryKey: mcpKeyAccessRoot(workspaceId),
      })
    },
  })
}
