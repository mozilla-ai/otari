import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  endpointHost,
  UPDATE_CHECK_DOC,
} from "@/features/updateCheck/useOperatorUpdateCheck"
import { useAnswerUpdateCheck, useUpdateCheck } from "@/shared/api/updateCheck"
import { docsSourceHref } from "@/shared/helpers/docs"
import { formatDateTime } from "@/shared/helpers/format"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * Whether this deployment checks otari.ai for new releases and security fixes,
 * and exactly what each check sends.
 *
 * Like `MaintenanceModeCard`, this is stored deployment state rather than one
 * of the config rows above it, except where config answers: then the switch
 * shows that answer and is locked, because the dashboard cannot change it.
 */
export function UpdateCheckCard() {
  const isHosted = useSurfaces()("update_check")
  const status = useUpdateCheck(isHosted)
  const answer = useAnswerUpdateCheck()

  if (!isHosted) {
    return null
  }

  const data = status.data
  const loading = status.isPending && !data
  const isLocked = data?.answered_by === "config"

  return (
    <SettingsGroup title="Update check">
      <div className="flex flex-col gap-4 py-4">
        <ErrorBanner error={status.error ?? answer.error} />
        {loading ? <PageLoading label="Loading the update check…" /> : null}
        {data ? (
          <>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-body">
                  Get notified when a new Otari release is available
                </p>
                <p className="mt-1 max-w-3xl text-caption">
                  {!data.checkable
                    ? `This is a development build (${data.current_version}), which is never checked.`
                    : isLocked
                      ? "Decided by this deployment's configuration (update_check, OTARI_UPDATE_CHECK, DO_NOT_TRACK or CI). Change it there."
                      : `Once a day, Otari asks ${endpointHost(data.endpoint)} whether a newer release or a security fix for this version exists, and shows it here and in the logs.`}
                </p>
              </div>
              <Toggle
                isSelected={data.enabled === true}
                onChange={(next) => answer.mutate({ enabled: next })}
                label="Get notified when a new Otari release is available"
                isDisabled={isLocked || answer.isPending}
              />
            </div>
            <dl className="grid max-w-3xl grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-caption">
              <dt>Sends</dt>
              <dd className="min-w-0 break-all">
                version <code>{data.current_version}</code>, install ID{" "}
                <code>{data.install_id}</code>{" "}
                <a
                  href={docsSourceHref(UPDATE_CHECK_DOC, "what-is-sent")}
                  target="_blank"
                  rel="noreferrer"
                  className="underline underline-offset-2"
                >
                  (what&rsquo;s sent)
                </a>
              </dd>
              <dt>To</dt>
              <dd className="min-w-0 break-all">
                <code>{data.endpoint}</code>
              </dd>
              {data.enabled === true ? (
                <>
                  <dt>Last checked</dt>
                  <dd>
                    {data.last_checked_at
                      ? formatDateTime(data.last_checked_at)
                      : "Not yet"}
                  </dd>
                  <dt>Latest release</dt>
                  <dd>
                    {data.latest_version ?? "Unknown"}
                    {data.update_available ? " (update available)" : ""}
                  </dd>
                </>
              ) : null}
            </dl>
          </>
        ) : null}
      </div>
    </SettingsGroup>
  )
}
