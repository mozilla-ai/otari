import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  BuiltInGuardrailCatalog,
  CreateOrganizationGuardrailDefinitionRequest,
  OrganizationGuardrailDefinition,
  UpdateOrganizationGuardrailDefinitionRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  BUILTIN_GUARDRAIL_CATALOG,
  ORGANIZATION_GUARDRAIL_DEFINITIONS,
} from "@/shared/api/queryKeys"

// The mandates and the remote profile list stay in `tools.ts`, where the card
// that reads them lives. These are what the organization guardrails page adds.

const DEFINITIONS = "/organizations/me/guardrail-definitions"

// The guardrails this gateway can build itself. It changes only when the
// deployment installs another any-guardrail, so nothing a user does in the
// dashboard invalidates it and a long window costs nothing.
export function useBuiltInGuardrailCatalog(enabled = true) {
  return useQuery({
    queryKey: [BUILTIN_GUARDRAIL_CATALOG],
    queryFn: () =>
      apiFetch<BuiltInGuardrailCatalog>("/tool-settings/guardrails/catalog"),
    staleTime: Number.POSITIVE_INFINITY,
    enabled,
  })
}

// One page is the whole list: the gateway refuses an eleventh definition
// (`MAX_DEFINITIONS_PER_ORGANIZATION`), and the route's default page is 100.
export function useOrganizationGuardrailDefinitions(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_GUARDRAIL_DEFINITIONS],
    queryFn: async () =>
      (await apiFetch<{ data: OrganizationGuardrailDefinition[] }>(DEFINITIONS))
        .data,
    staleTime: 60_000,
    enabled,
  })
}

function useInvalidateDefinitions() {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({
      queryKey: [ORGANIZATION_GUARDRAIL_DEFINITIONS],
    })
  }
}

export function useCreateOrganizationGuardrailDefinition() {
  const onSuccess = useInvalidateDefinitions()
  return useMutation({
    mutationFn: (body: CreateOrganizationGuardrailDefinitionRequest) =>
      apiFetch<OrganizationGuardrailDefinition>(DEFINITIONS, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess,
  })
}

export function useUpdateOrganizationGuardrailDefinition() {
  const onSuccess = useInvalidateDefinitions()
  return useMutation({
    mutationFn: ({
      definitionId,
      body,
    }: {
      definitionId: string
      body: UpdateOrganizationGuardrailDefinitionRequest
    }) =>
      apiFetch<OrganizationGuardrailDefinition>(
        `${DEFINITIONS}/${encodeURIComponent(definitionId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess,
  })
}

// Refused with a 409 naming the profiles while a mandate still points here; the
// caller reads that message, so nothing here special-cases it.
export function useDeleteOrganizationGuardrailDefinition() {
  const onSuccess = useInvalidateDefinitions()
  return useMutation({
    mutationFn: (definitionId: string) =>
      apiFetch<{ message: string }>(
        `${DEFINITIONS}/${encodeURIComponent(definitionId)}`,
        { method: "DELETE" },
      ),
    onSuccess,
  })
}
