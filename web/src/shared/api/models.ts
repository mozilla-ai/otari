import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CatalogModelDetail,
  CatalogQueryParams,
  CatalogResponse,
  DiscoverableModelsResponse,
  ModelListResponse,
  ModelMetadataOverride,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  CATALOG,
  DISCOVERABLE,
  MODELS,
  NO_RETRY,
  workspaceMeta,
} from "@/shared/api/queryKeys"

// The catalog folded by model, priced for the caller. Any session may read it,
// like `/v1/models`; the detail is keyed under the list so a pricing write that
// invalidates CATALOG takes every open detail with it.
// Keyed beside the model id so a detail read and a list read never share a
// cache entry (a model whose id is `list` included), while both still fall
// under the CATALOG prefix invalidations use.
// Existing consumers such as the Playground keep their original bounded read.
// The Models page supplies its filters, skip, and limit explicitly.
export function useCatalog(
  params: CatalogQueryParams = { limit: 1000 },
  enabled = true,
) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value == null) continue
    for (const entry of Array.isArray(value) ? value : [value]) {
      search.append(key, String(entry))
    }
  }
  return useQuery({
    ...NO_RETRY,
    queryKey: [CATALOG, "list", params],
    meta: workspaceMeta(params.workspace_id),
    queryFn: ({ signal }) =>
      apiFetch<CatalogResponse>(`/catalog/models?${search}`, { signal }),
    staleTime: 60_000,
    // The previous page stands in while a filter or a page changes, never
    // across workspaces: another workspace's list names models this one may
    // not call.
    placeholderData: (previous, previousQuery) =>
      (previousQuery?.queryKey[2] as CatalogQueryParams | undefined)
        ?.workspace_id === params.workspace_id
        ? previous
        : undefined,
    enabled,
  })
}

/**
 * The catalog narrowed to a search term, for a picker.
 *
 * The term goes to the server, so what comes back is the matches out of the
 * whole catalog rather than the matches out of whatever page was fetched
 * (otari#1380). Bounded because a picker shows a handful of rows: the popover
 * cannot render a thousand, and `count` is what tells the reader how many more
 * there are.
 *
 * Debounce the term before it gets here (`useDebounced`), or every keystroke is
 * a request and the answers race.
 *
 * `workspaceId` narrows the matches to what a request billed to that workspace
 * may call; empty is the caller's whole reach, and `null` holds the read until
 * the workspace is known (`useNarrowingWorkspace`).
 */
export function useCatalogSearch(
  search: string,
  limit: number,
  enabled = true,
  workspaceId: string | null = "",
) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [CATALOG, "search", search, limit, workspaceId ?? ""],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<CatalogResponse>(
        `/catalog/models?limit=${limit}${search ? `&search=${encodeURIComponent(search)}` : ""}${workspaceId ? `&workspace_id=${encodeURIComponent(workspaceId)}` : ""}`,
      ),
    staleTime: 60_000,
    // Never across workspaces, for the reason `useCatalog` gives.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[4] === (workspaceId ?? "") ? previous : undefined,
    enabled: enabled && workspaceId !== null,
  })
}

/**
 * One model, as the caller may use it. `workspaceId` narrows its offerings to
 * what a request billed to that workspace may call; empty is the caller's whole
 * reach. `enabled` holds the read until the workspace is known.
 */
export function useCatalogModel(
  modelId: string | undefined,
  workspaceId = "",
  enabled = true,
) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [CATALOG, "model", modelId, workspaceId],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      // Segment by segment: the id carries its vendor, `z-ai/glm-5.3`, and
      // the slash is the path's, not the id's to encode.
      apiFetch<CatalogModelDetail>(
        `/catalog/models/${(modelId ?? "")
          .split("/")
          .map(encodeURIComponent)
          .join(
            "/",
          )}${workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : ""}`,
      ),
    staleTime: 60_000,
    enabled: modelId !== undefined && enabled,
  })
}

// `enabled` is not the operator composition the reads below use: the catalog is
// readable by any signed-in caller and is already narrowed server-side to what
// that caller could route to. It is for a page with nothing to ask about yet,
// such as an overview before a workspace is selected. `workspaceId` narrows it
// to what a request billed to that workspace may call.
export function useModels(enabled = true, workspaceId = "") {
  return useQuery({
    ...NO_RETRY,
    queryKey: [MODELS, workspaceId],
    meta: workspaceMeta(workspaceId),
    queryFn: () =>
      apiFetch<ModelListResponse>(
        workspaceId
          ? `/models?workspace_id=${encodeURIComponent(workspaceId)}`
          : "/models",
      ),
    staleTime: 60_000,
    enabled,
  })
}

export function useDiscoverableModels(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [DISCOVERABLE],
    queryFn: () => apiFetch<DiscoverableModelsResponse>("/models/discoverable"),
    staleTime: 5 * 60_000,
    enabled,
  })
}

// Deliberately unscoped, and deliberately without the parameter the endpoint
// would accept: the gateway stores every alias in the default workspace because
// resolution reads a process-wide name-keyed cache, so a filtered list would
// hide live aliases. Leaving the argument here would be a loaded gun beside the
// comment explaining why it must not be fired.
//
// `enabled` is the `useToolSettings` composition: the read is
// deployment-operator-only, so a tenant-facing page declines to ask.

// What a model's server and models.dev do not say, set by whoever holds the
// key that serves it: the operator for the deployment's own providers, an
// organization's owners and admins for its own keys. Saving every field empty
// clears the override. Both the catalog and `/v1/models` read it.
export type ModelOverrideOwner = "deployment" | "organization"

const overridePath = (owner: ModelOverrideOwner, selector: string) =>
  `${owner === "organization" ? "/organizations/me" : ""}/model-overrides/${encodeURIComponent(selector)}`

export function useSetModelOverride() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      owner,
      selector,
      body,
    }: {
      owner: ModelOverrideOwner
      selector: string
      body: ModelMetadataOverride
    }) =>
      apiFetch<ModelMetadataOverride>(overridePath(owner, selector), {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [CATALOG] })
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
    },
  })
}
