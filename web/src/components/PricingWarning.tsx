import { Button } from "@heroui/react";
import { useState } from "react";

import { useSettings, useUpdateSettings } from "@/api/hooks";
import { InfoBanner } from "@/components/ui";

// A gateway-wide alarm, shown on every management page: when `require_pricing` is
// on but `default_pricing` is off, every request for an unpriced model is being
// rejected (402). It lives in the app shell rather than any one page so an
// operator sees it regardless of where they are, or how the state arose (e.g.
// flipping require_pricing on via config long after onboarding). Dismissible per
// tab; it reappears on reload while the condition holds.
export function PricingWarning() {
  const settings = useSettings();
  const updateSettings = useUpdateSettings();
  const [dismissed, setDismissed] = useState(false);

  const needsPricing = settings.data?.require_pricing === true && settings.data.default_pricing === false;
  if (!needsPricing || dismissed) {
    return null;
  }

  return (
    <div className="shrink-0 px-6 pt-3">
      <InfoBanner tone="warning">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span>
            Requests are rejected until pricing is set (<code>require_pricing</code> is on). Enable default pricing to
            meter new models with public rates right away.
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
