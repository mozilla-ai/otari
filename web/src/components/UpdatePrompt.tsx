import { Button } from "@heroui/react";
import { useRef, useState } from "react";

import { useDashboardBuild } from "@/api/hooks";

// True once the gateway starts serving a different bundle than the one this tab
// loaded. The comparison is against the first build this tab ever saw rather
// than a value baked in at build time: "the app changed while you had it open"
// is the question worth asking, and a tab opened after a deploy is already
// current and must stay quiet.
export function useUpdateAvailable(): boolean {
  const { data } = useDashboardBuild();
  const loadedBuild = useRef<string | null>(null);

  if (data && loadedBuild.current === null) {
    loadedBuild.current = data.build;
  }
  return data != null && loadedBuild.current != null && data.build !== loadedBuild.current;
}

// Offers the reload that picks up a new build. Deliberately not a modal: an
// operator part-way through pricing a model should not be interrupted, and the
// stale tab keeps working until they choose.
export function UpdatePrompt() {
  const updateAvailable = useUpdateAvailable();
  const [dismissed, setDismissed] = useState(false);

  if (!updateAvailable || dismissed) {
    return null;
  }

  return (
    <div
      role="status"
      className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-[var(--otari-brand)] bg-[var(--otari-brand-tint)] px-4 py-3 text-sm text-[var(--otari-brand-dark)]"
    >
      <span className="flex-1">
        <strong className="font-semibold">An update is available.</strong> This tab is running an older version of the
        dashboard. Reloading keeps you signed in.
      </span>
      <span className="flex items-center gap-2">
        {/* A plain reload is enough: the gateway serves index.html with
            no-store, so this fetches the new bundle rather than the cached one,
            and the master key lives in sessionStorage, which survives it. */}
        <Button size="sm" variant="primary" onPress={() => window.location.reload()}>
          Update now
        </Button>
        <Button size="sm" variant="ghost" onPress={() => setDismissed(true)}>
          Later
        </Button>
      </span>
    </div>
  );
}
