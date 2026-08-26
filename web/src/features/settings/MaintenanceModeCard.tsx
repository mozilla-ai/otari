import { Card } from "@heroui/react"
import { useMaintenanceMode, useSetMaintenanceMode } from "@/shared/api/hooks"
import { ErrorBanner, PageLoading } from "@/shared/components/ui"

import { Toggle } from "./Toggle"

/**
 * The sign-in freeze: stop new dashboard sessions while the gateway redeploys.
 *
 * Deliberately not one of the config rows above it on this page. Those project
 * `GatewayConfig` fields and are applied to the worker that served the save;
 * this is deployment state with no config field behind it, read from its stored
 * row on every sign-in attempt so a deployment running more than one replica
 * freezes all of them at once. The wording says what it does and, just as
 * importantly, what it does not: an operator about to flip this needs to know
 * their own session survives it and that the API keeps serving.
 */
export function MaintenanceModeCard() {
  const maintenance = useMaintenanceMode()
  const setMaintenance = useSetMaintenanceMode()

  const data = maintenance.data
  const enabled = data?.enabled ?? false
  // Nothing is claimed about the freeze until the server has answered. Showing
  // an "off" switch while the request is in flight would state the opposite of
  // the truth on exactly the deployment an operator is checking on.
  const loading = maintenance.isPending && !data

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-title">Maintenance mode</h2>
      <Card>
        <Card.Content className="flex flex-col divide-y divide-border px-5 py-1">
          <div className="flex flex-col gap-4 py-4">
            <ErrorBanner error={maintenance.error ?? setMaintenance.error} />
            {loading ? <PageLoading label="Loading maintenance mode…" /> : null}
            {data ? (
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground">
                    Freeze new dashboard sign-ins
                  </p>
                  <p className="mt-1 max-w-3xl text-sm text-muted">
                    {enabled
                      ? "Nobody can start a new dashboard session. Sessions already open keep working, and the API and management endpoints still answer the master key, so you can turn this back off from here or with your key."
                      : "Turn this on before a redeploy so nobody signs in mid-migration. Sessions already open keep working, and it does not touch the API: keys and completions carry on serving. Keep your master key to hand: it is what lifts the freeze once your own session is gone."}
                  </p>
                </div>
                <Toggle
                  checked={enabled}
                  onChange={(next) => setMaintenance.mutate(next)}
                  label="Freeze new dashboard sign-ins"
                  disabled={setMaintenance.isPending}
                />
              </div>
            ) : null}
          </div>
        </Card.Content>
      </Card>
    </section>
  )
}
