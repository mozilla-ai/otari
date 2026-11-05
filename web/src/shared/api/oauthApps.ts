import { useMutation, useQuery } from "@tanstack/react-query"
import type { AppAuthorizationResponse, AppDetails } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { OAUTH_APPS } from "@/shared/api/queryKeys"

/** Resolve the registered app before displaying consent or following a redirect. */
export function useOAuthApp(clientId: string, redirectUri: string) {
  return useQuery({
    queryKey: [OAUTH_APPS, clientId, redirectUri],
    queryFn: () =>
      apiFetch<AppDetails>(
        `/oauth/apps/clients/${encodeURIComponent(clientId)}?redirect_uri=${encodeURIComponent(redirectUri)}`,
      ),
    retry: false,
  })
}

/** The server validates all OAuth parameters before issuing a one-use grant. */
export function useAuthorizeOAuthApp(params: URLSearchParams) {
  return useMutation({
    mutationFn: (workspaceId: string) =>
      apiFetch<AppAuthorizationResponse>("/oauth/apps/authorize", {
        method: "POST",
        body: JSON.stringify({
          client_id: params.get("client_id"),
          redirect_uri: params.get("redirect_uri"),
          response_type: params.get("response_type"),
          state: params.get("state"),
          code_challenge: params.get("code_challenge"),
          code_challenge_method: params.get("code_challenge_method"),
          workspace_id: workspaceId,
        }),
      }),
    onSuccess: (result) => window.location.assign(result.redirect_uri),
  })
}
