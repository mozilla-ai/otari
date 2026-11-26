import { useMutation, useQuery } from "@tanstack/react-query"
import type {
  AppAuthorizationRequest,
  AppAuthorizationResponse,
  AppDetails,
  KeyScope,
  Workspaces,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { OAUTH_APPS } from "@/shared/api/queryKeys"

/** Resolve the registered app before displaying consent or following a redirect. */
export function useOAuthApp(clientId: string, redirectUri: string) {
  return useQuery({
    queryKey: [OAUTH_APPS, "client", clientId, redirectUri],
    queryFn: () =>
      apiFetch<AppDetails>(
        `/oauth/apps/clients?${new URLSearchParams({ client_id: clientId, redirect_uri: redirectUri })}`,
      ),
    retry: false,
  })
}

/** Search only workspaces where the signed-in user can approve app access. */
export function useOAuthAppWorkspaces(search: string, enabled: boolean) {
  return useQuery({
    queryKey: [OAUTH_APPS, "workspaces", search],
    queryFn: () =>
      apiFetch<Workspaces>(
        `/oauth/apps/workspaces?${new URLSearchParams({ search, limit: "50" })}`,
      ),
    placeholderData: (previous) => previous,
    enabled,
  })
}

/** What the person chose on the consent screen besides the app's own parameters. */
export interface AppApproval {
  /** Absent for an app given an access token, which bills no workspace. */
  workspaceId?: string
  /** The organization an access token reads; absent for an app given a key. */
  organizationId?: string
  /** Blank sets no spending limit. */
  limit: string
  expiration: NonNullable<AppAuthorizationRequest["expiration"]>
}

/** The server validates all OAuth parameters before issuing a one-use grant. */
export function useAuthorizeOAuthApp(
  params: URLSearchParams,
  scopes: readonly KeyScope[],
) {
  return useMutation({
    mutationFn: ({
      workspaceId,
      organizationId,
      limit,
      expiration,
    }: AppApproval) =>
      apiFetch<AppAuthorizationResponse>("/oauth/apps/authorize", {
        method: "POST",
        body: JSON.stringify({
          client_id: params.get("client_id"),
          redirect_uri: params.get("redirect_uri"),
          response_type: params.get("response_type") ?? undefined,
          state: params.get("state"),
          code_challenge: params.get("code_challenge"),
          code_challenge_method: params.get("code_challenge_method"),
          ...(workspaceId ? { workspace_id: workspaceId } : {}),
          ...(organizationId ? { organization_id: organizationId } : {}),
          scopes,
          ...(limit ? { limit } : {}),
          expiration,
        }),
      }),
    onSuccess: (result) => window.location.assign(result.redirect_uri),
  })
}
