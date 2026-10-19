import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  HealthCheckDisabled,
  HealthCheckSettings,
  HealthCheckStatus,
  UpdateHealthCheckSettings,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { HEALTH_CHECKS, HEALTH_CHECKS_POLL_MS } from "@/shared/api/queryKeys"

/**
 * The Health page's one read: the settings, the run in progress and the
 * latest finished run. Polled, because a run is started and finished by a
 * worker, not by anything this tab does.
 */
export function useHealthCheckStatus() {
  return useQuery({
    queryKey: [HEALTH_CHECKS],
    queryFn: () => apiFetch<HealthCheckStatus>("/health-checks"),
    refetchInterval: HEALTH_CHECKS_POLL_MS,
    refetchOnWindowFocus: true,
    staleTime: 0,
  })
}

// Both writes answer with the settings, which are folded into the status read
// rather than refetching it.
function useSettingsWrite<Variables>(
  mutationFn: (variables: Variables) => Promise<HealthCheckSettings>,
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn,
    onSuccess: (settings) => {
      queryClient.setQueryData<HealthCheckStatus>([HEALTH_CHECKS], (status) =>
        status ? { ...status, settings } : status,
      )
    },
  })
}

export function useUpdateHealthCheckSettings() {
  return useSettingsWrite((body: UpdateHealthCheckSettings) =>
    apiFetch<HealthCheckSettings>("/health-checks/settings", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  )
}

/** Ask for a run now. A worker starts it within one poll; the status poll shows it. */
export function useRunHealthChecks() {
  return useSettingsWrite(() =>
    apiFetch<HealthCheckSettings>("/health-checks/run", { method: "POST" }),
  )
}

/**
 * Switch off every model the latest run found failing, on its hosted provider.
 * The status is read again, so the models leave the failing list at once.
 */
export function useDisableFailingModels() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<HealthCheckDisabled>("/health-checks/disable-failing", {
        method: "POST",
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: [HEALTH_CHECKS] }),
  })
}
