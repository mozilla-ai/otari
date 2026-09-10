import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  GatewaySettings,
  MailSettings,
  MaintenanceMode,
  SendTestMailRequest,
  SendTestMailResponse,
  UpdateSettingsRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import {
  DISCOVERABLE,
  MAIL_SETTINGS,
  MAINTENANCE_MODE,
  MAINTENANCE_MODE_POLL_MS,
  MODELS,
  SETTINGS,
} from "@/shared/api/queryKeys"

export function useSettings(enabled = true) {
  return useQuery({
    queryKey: [SETTINGS],
    queryFn: () => apiFetch<GatewaySettings>("/settings"),
    staleTime: 60_000,
    enabled,
  })
}

export function useUpdateSettings() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: UpdateSettingsRequest) =>
      apiFetch<GatewaySettings>("/settings", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData([SETTINGS], data)
      // Toggling discovery changes which models the catalog and picker report.
      void queryClient.invalidateQueries({ queryKey: [MODELS] })
      void queryClient.invalidateQueries({ queryKey: [DISCOVERABLE] })
    },
  })
}

/**
 * Whether this deployment is refusing new dashboard sign-ins.
 *
 * Not read from the bootstrap, which carries the same flag: that one is fetched
 * once per page load and cached for the life of the tab, which is right for the
 * sign-in screen (it renders before there is anything to poll with) and wrong
 * for the switch that changes it. This is the live value the card renders.
 */
export function useMaintenanceMode() {
  return useQuery({
    queryKey: [MAINTENANCE_MODE],
    queryFn: () => apiFetch<MaintenanceMode>("/settings/maintenance-mode"),
    // Polled and refreshed on focus, unlike every other settings read here.
    // A `staleTime` alone schedules nothing, and this app turns
    // `refetchOnWindowFocus` off globally, so a card left open would keep
    // showing whatever it fetched on mount. That is the one wrong answer this
    // card can give: another operator or an API client can flip the freeze, and
    // reporting a deployment open when it is frozen (or frozen when it is back)
    // is worse than a moment's blank. Same treatment as `useDashboardBuild`,
    // for the same reason: the value changes underneath the tab.
    refetchInterval: MAINTENANCE_MODE_POLL_MS,
    refetchOnWindowFocus: true,
    staleTime: 0,
  })
}

/**
 * Freeze or unfreeze dashboard sign-ins.
 *
 * Nothing else is invalidated: the freeze changes no data any other page shows,
 * and it deliberately does not touch the caller's own session, so the tab that
 * flipped it keeps working either way.
 */
export function useSetMaintenanceMode() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (enabled: boolean) =>
      apiFetch<MaintenanceMode>("/settings/maintenance-mode", {
        method: "PATCH",
        body: JSON.stringify({ enabled }),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData([MAINTENANCE_MODE], data)
    },
  })
}

export function useMailSettings() {
  return useQuery({
    queryKey: [MAIL_SETTINGS],
    queryFn: () => apiFetch<MailSettings>("/settings/mail"),
    staleTime: 60_000,
  })
}

/**
 * Sends a real message to prove the transport works.
 *
 * Nothing is invalidated on success: a test send changes no server state, and
 * the outcome lives in the mutation's own result. A failure comes back two
 * ways, and the page distinguishes them: a 200 with `ok: false` is a configured
 * transport that refused, while a 503 is a deployment with no transport at all.
 */
export function useSendTestMail() {
  return useMutation({
    mutationFn: (body: SendTestMailRequest) =>
      apiFetch<SendTestMailResponse>("/settings/mail/test", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  })
}

// Any signed-in caller may read this now (otari-ai#1969); a non-operator is
// answered without the three service-endpoint fields rather than refused, so a
// page renders whatever came back instead of gating the read. `enabled` stays
// for the callers that have a reason not to ask at all.
