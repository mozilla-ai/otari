import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type {
  AlertRule,
  AlertRuleTestResult,
  CreateAlertRuleRequest,
  UpdateAlertRuleRequest,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { ORGANIZATION_ALERT_RULES } from "@/shared/api/queryKeys"

// Where an organization's budget alerts go. One module rather than more rows in
// `organizations.ts`, matching the one-module-per-domain rule: the rules are
// their own page and share no cache with the roster or the domain claims.
//
// The destination is write-only end to end. `AlertRule.destination` is the
// server's redaction (scheme and host, everything else masked), so a form that
// loaded a rule and submitted it whole would send the mask back as a real
// value. `UpdateAlertRuleRequest.destination` is therefore omitted rather than
// echoed whenever the operator has not typed a new one, which is what the
// server's "omit to keep the stored destination" contract expects.

const ALERT_RULES_PATH = "/v1/organizations/me/alert-rules"

export function useAlertRules(enabled = true) {
  return useQuery({
    queryKey: [ORGANIZATION_ALERT_RULES],
    queryFn: () =>
      apiFetch<{ data: AlertRule[]; count: number }>(ALERT_RULES_PATH),
    staleTime: 60_000,
    enabled,
  })
}

function invalidateAlertRules(queryClient: ReturnType<typeof useQueryClient>) {
  void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_ALERT_RULES] })
}

export function useCreateAlertRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateAlertRuleRequest) =>
      apiFetch<AlertRule>(ALERT_RULES_PATH, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateAlertRules(queryClient),
  })
}

export function useUpdateAlertRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      ruleId,
      body,
    }: {
      ruleId: string
      body: UpdateAlertRuleRequest
    }) =>
      apiFetch<AlertRule>(`${ALERT_RULES_PATH}/${encodeURIComponent(ruleId)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateAlertRules(queryClient),
  })
}

export function useDeleteAlertRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (ruleId: string) =>
      apiFetch<{ message: string }>(
        `${ALERT_RULES_PATH}/${encodeURIComponent(ruleId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => invalidateAlertRules(queryClient),
  })
}

/**
 * Send a sample alert to one rule's destination now.
 *
 * Deliberately does not invalidate the list: a test writes no delivery row and
 * changes nothing about the rule, so refetching would only make the table
 * flicker. The outcome is returned to the caller instead, which is what the
 * page renders beside the row.
 */
export function useTestAlertRule() {
  return useMutation({
    mutationFn: (ruleId: string) =>
      apiFetch<AlertRuleTestResult>(
        `${ALERT_RULES_PATH}/${encodeURIComponent(ruleId)}/test`,
        { method: "POST" },
      ),
  })
}
