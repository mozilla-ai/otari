import { useQuery } from "@tanstack/react-query"
import type {
  DiscoverableModelsResponse,
  ModelListResponse,
  ModelMetadataResponse,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  DISCOVERABLE,
  METADATA,
  MODELS,
  NO_RETRY,
} from "@/shared/api/queryKeys"

export function useModels() {
  return useQuery({
    ...NO_RETRY,
    queryKey: [MODELS],
    queryFn: () => apiFetch<ModelListResponse>("/v1/models"),
    staleTime: 60_000,
  })
}

export function useDiscoverableModels(enabled = true) {
  return useQuery({
    ...NO_RETRY,
    queryKey: [DISCOVERABLE],
    queryFn: () =>
      apiFetch<DiscoverableModelsResponse>("/v1/models/discoverable"),
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
    queryFn: () => apiFetch<ModelMetadataResponse>("/v1/models/metadata"),
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
