import { useQuery } from "@tanstack/react-query"
import type {
  CatalogModelDetail,
  CatalogResponse,
  DiscoverableModelsResponse,
  ModelListResponse,
  ModelMetadataResponse,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  CATALOG,
  DISCOVERABLE,
  METADATA,
  MODELS,
  NO_RETRY,
} from "@/shared/api/queryKeys"

// The catalog folded by model, priced for the caller. Any session may read it,
// like `/v1/models`; the detail is keyed under the list so a pricing write that
// invalidates CATALOG takes every open detail with it.
// Keyed beside the model id so a detail read and a list read never share a
// cache entry, while both still fall under the CATALOG prefix invalidations use.
export function useCatalog() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [CATALOG, "list"],
    queryFn: () => apiFetch<CatalogResponse>("/catalog/models"),
    staleTime: 60_000,
  })
}

export function useCatalogModel(modelId: string | undefined) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [CATALOG, modelId],
    queryFn: () =>
      // Segment by segment: the id carries its vendor, `z-ai/glm-5.3`, and
      // the slash is the path's, not the id's to encode.
      apiFetch<CatalogModelDetail>(
        `/catalog/models/${(modelId ?? "")
          .split("/")
          .map(encodeURIComponent)
          .join("/")}`,
      ),
    staleTime: 60_000,
    enabled: modelId !== undefined,
  })
}

// `enabled` is not the operator composition the reads below use: the catalog is
// readable by any signed-in caller and is already narrowed server-side to what
// that caller could route to. It is for a page with nothing to ask about yet,
// such as an overview before a workspace is selected.
export function useModels(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [MODELS],
    queryFn: () => apiFetch<ModelListResponse>("/models"),
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

// Static metadata for every configured provider: capabilities, doc and pricing
// links, display name. Network-free gateway-side (bundled datasets), so it does
// not move within a session; kept fresh for a few minutes like discovery.
//
// `enabled` is for the same composition `useBudgets` documents: this read is
// deployment-wide and answers 403 to anyone who does not operate the deployment
// (#821), so a tenant-facing page declines to ask rather than surfacing the
// refusal (otari#838).

export function useModelMetadata(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [METADATA],
    queryFn: () => apiFetch<ModelMetadataResponse>("/models/metadata"),
    staleTime: 10 * 60_000,
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
