import { Button } from "@heroui/react";
import { useState } from "react";
import { Link } from "@tanstack/react-router";

import { useFailureCount, useSettings, useUpdateSettings } from "@/api/hooks";
import { InfoBanner } from "@/components/ui";
import { HOUR_S } from "@/lib/timeRange";

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
  const settings = useSettings();
  const updateSettings = useUpdateSettings();
  const [dismissed, setDismissed] = useState(false);

  const needsPricing = settings.data?.require_pricing === true && settings.data.default_pricing === false;
  const showing = needsPricing && !dismissed;

  // Every failure class the gateway served is counted (402 no pricing, 403 budget
  // or model access, 502 provider), not only the pricing rejections: the operator's
  // question in this state is "is traffic getting through", and over-reporting a
  // failure is safer than a banner reading "0" while requests are being dropped.
  // Imported usage is excluded, so the link's filtered view matches this count.
  // Only polled while the alarm is up.
  const failures = useFailureCount(HOUR_S, showing);
  const failureCount = failures.data?.total ?? 0;

  if (!showing) {
    return null;
  }

  return (
    <div className="shrink-0 px-6 pt-3">
      <InfoBanner tone="warning">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span>
            Requests are rejected until pricing is set (<code>require_pricing</code> is on). Enable default pricing to
            meter new models with public rates right away.
            {failureCount > 0 ? (
              <>
                {" "}
                <strong className="font-semibold">
                  {failureCount.toLocaleString()} {failureCount === 1 ? "request" : "requests"} failed in the last hour.
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
              {updateSettings.isPending ? "Enabling…" : "Enable default pricing"}
            </Button>
            <Button size="sm" variant="ghost" onPress={() => setDismissed(true)}>
              Dismiss
            </Button>
          </span>
        </div>
      </InfoBanner>
    </div>
  );
}
