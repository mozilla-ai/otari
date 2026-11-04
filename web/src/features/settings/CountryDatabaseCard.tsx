import { useState } from "react"
import { FileTrigger } from "react-aria-components"
import { FiUpload } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  useCountryDatabase,
  useRemoveCountryDatabase,
  useUploadCountryDatabase,
} from "@/shared/api/settings"
import {
  formatDate,
  formatDateTime,
  formatFileSize,
  formatRelative,
} from "@/shared/helpers/format"

/**
 * The country file a request's address is looked up in, for the provider
 * country gate: uploaded here and stored in the database, so every replica
 * reads the same one without a file mounted into each container.
 */
export function CountryDatabaseCard() {
  const status = useCountryDatabase()
  const upload = useUploadCountryDatabase()
  const remove = useRemoveCountryDatabase()
  const [confirmingRemove, setConfirmingRemove] = useState(false)

  const data = status.data
  const stored = data?.stored ?? null
  const loading = status.isPending && !data

  return (
    <SettingsGroup title="Country file">
      <div className="flex flex-col gap-4 py-4">
        <ErrorBanner error={status.error ?? upload.error} />
        {loading ? <PageLoading label="Loading the country file…" /> : null}
        {data?.client_country_database_set ? (
          <InfoBanner>
            <code>client_country_database</code> names a file in the
            gateway&rsquo;s configuration, and that file is read. A file stored
            here is kept but not read until the setting is removed.
          </InfoBanner>
        ) : null}
        {data ? (
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <p className="text-body">
                {stored ? stored.database_type : "No country file stored"}
              </p>
              <p className="mt-1 max-w-3xl text-caption">
                A MaxMind DB country file, such as GeoLite2 Country or DB-IP
                Lite Country, in which each request&rsquo;s address is looked up
                for the providers that serve only some countries. Every replica
                reads the stored file within 30 seconds of an upload.
              </p>
              {stored ? (
                <dl className="mt-3 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[10rem_1fr]">
                  <dt className="text-muted">Built</dt>
                  <dd className="text-foreground">
                    {formatDate(stored.built_at)} (
                    {formatRelative(stored.built_at)})
                  </dd>
                  <dt className="text-muted">Size</dt>
                  <dd className="text-foreground tabular-nums">
                    {formatFileSize(stored.size_bytes)}
                  </dd>
                  <dt className="text-muted">Uploaded</dt>
                  <dd className="text-foreground">
                    {formatDateTime(stored.uploaded_at)}
                  </dd>
                </dl>
              ) : null}
            </div>
            <div className="flex shrink-0 flex-wrap gap-2">
              <FileTrigger
                acceptedFileTypes={[".mmdb"]}
                onSelect={(files) => {
                  const file = files?.[0]
                  if (file) upload.mutate(file)
                }}
              >
                <Button
                  size="sm"
                  variant="primary"
                  isPending={upload.isPending}
                >
                  <FiUpload aria-hidden className="size-4" />
                  {stored ? "Replace file" : "Upload file"}
                </Button>
              </FileTrigger>
              {stored ? (
                <Button size="sm" onPress={() => setConfirmingRemove(true)}>
                  Remove
                </Button>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
      <ConfirmDialog
        isOpen={confirmingRemove}
        onOpenChange={(open) => {
          if (open) return
          setConfirmingRemove(false)
          remove.reset()
        }}
        heading="Remove the country file"
        body="Requests are no longer looked up in it. While a provider that serves only some countries has no other country source, the file cannot be removed."
        confirmLabel="Remove file"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() =>
          remove.mutate(undefined, {
            onSuccess: () => setConfirmingRemove(false),
          })
        }
      />
    </SettingsGroup>
  )
}
