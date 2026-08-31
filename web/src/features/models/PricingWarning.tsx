import { Button } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { useState } from "react"

import { isDeploymentOperator } from "@/features/organization/roles"
import {
  useFailureCount,
  useOrganizationContext,
  useSettings,
  useUpdateSettings,
} from "@/shared/api/hooks"
import { InfoBanner } from "@/shared/components/ui"
import { HOUR_S } from "@/shared/helpers/timeRange"

// A gateway-wide alarm, shown on every management page: when `require_pricing` is
// on but `default_pricing` is off, every request for an unpriced model is being
// rejected (402). It lives in the app shell rather than any one page so an
// operator sees it regardless of where they are, or how the state arose (e.g.
// flipping require_pricing on via config long after onboarding). Dismissible per
// tab; it reappears on reload while the condition holds.
//
// The banner also carries a live count of requests that failed in the last hour,
// so the reject state reads as an active incident (traffic is being dropped right
// now) rather than a static config note, and links into the activity log filtered
// to those failures.
export function PricingWarning() {
  // Both the read behind the alarm and the button that clears it are
  // deployment-operator-only (`require_deployment_operator` on `/v1/settings`),
  // so the audience is stated here rather than left to be inferred from a
  // refused query: without it every tenant page load fired a `GET /v1/settings`
  // that 403s to feed a banner that could never render for them (#834). Off the
  // organization context, which the shell reads anyway, for the reason
  // `useProviderKeyEncryption` does: a second request to ask the same question
  // is the cost this removes.
  const organization = useOrganizationContext()
  const isOperator = isDeploymentOperator(organization.data)
  const settings = useSettings(isOperator)
  const updateSettings = useUpdateSettings()
  const [dismissed, setDismissed] = useState(false)

  // The render is gated on the same answer the request is, because a disabled
  // query still serves whatever sits under its key and the query client outlives
  // a sign-out: without this, a session that follows an operator's in the same
  // tab renders their settings and a button that would be refused.
  const needsPricing =
    isOperator &&
    settings.data?.require_pricing === true &&
    settings.data.default_pricing === false
  const showing = needsPricing && !dismissed

  // Every failure class the gateway served is counted (402 no pricing, 403 budget
  // or model access, 502 provider), not only the pricing rejections: the operator's
  // question in this state is "is traffic getting through", and over-reporting a
  // failure is safer than a banner reading "0" while requests are being dropped.
  // Imported usage is excluded, so the link's filtered view matches this count.
  // Only polled while the alarm is up.
  const failures = useFailureCount(HOUR_S, showing)
  const failureCount = failures.data?.total ?? 0

  if (!showing) {
    return null
  }

  return (
    // Out of flow, pinned to the top of the shell: in flow it pushed the whole
    // shell down on every page the alarm is up on. Under the mobile drawer's z-40.
    <div className="absolute inset-x-0 top-0 z-30 px-6 pt-4">
      <InfoBanner tone="warning">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span>
            Requests are rejected until pricing is set (
            <code>require_pricing</code> is on). Enable default pricing to meter
            new models with public rates right away.
            {failureCount > 0 ? (
              <>
                {" "}
                <strong className="font-semibold">
                  {failureCount.toLocaleString()}{" "}
                  {failureCount === 1 ? "request" : "requests"} failed in the
                  last hour.
                </strong>{" "}
                <Link
                  to="/activity"
                  search={{ status: "error", range: "1h", source: "gateway" }}
                  className="underline underline-offset-2"
                >
                  View failed requests
                </Link>
              </>
            ) : null}
          </span>
          <span className="flex items-center gap-2">
            <Button
              size="sm"
              variant="primary"
              isDisabled={updateSettings.isPending}
              onPress={() => updateSettings.mutate({ default_pricing: true })}
            >
              {updateSettings.isPending
                ? "Enabling…"
                : "Enable default pricing"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onPress={() => setDismissed(true)}
            >
              Dismiss
            </Button>
          </span>
        </div>
      </InfoBanner>
    </div>
  )
}
