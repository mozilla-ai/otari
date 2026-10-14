import { useToolSettings } from "@/shared/api/tools"

/** Whether a guardrails service is configured for this gateway.
 *
 *  A policy guardrail is a request to a separate service (`guardrails_url`).
 *  With none configured there is nothing to call, so adding one would either
 *  refuse every request through the policy or do nothing.
 *
 *  Reads as configured while the settings load: a control that starts enabled
 *  and stays enabled beats one that flickers from disabled to enabled.
 */
export function useGuardrailsConfigured(enabled: boolean): boolean {
  const settings = useToolSettings(enabled)
  const field = settings.data?.fields.find(
    (entry) => entry.key === "guardrails_url",
  )
  const value = typeof field?.value === "string" ? field.value.trim() : ""
  return settings.isLoading || value !== ""
}
