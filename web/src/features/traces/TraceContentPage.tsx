import { type ReactNode, useState } from "react"
import type { ContentCaptureLevel, TraceSettings } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { TablePagination } from "@/design-system/data/TablePagination"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { Segmented } from "@/design-system/navigation/Segmented"
import { canManageWorkspace, memberLabel } from "@/features/organization/roles"
import { ContentReadsList } from "@/features/traces/ContentReadsList"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import {
  useContentAccessLog,
  usePurgeTraceContent,
  useTraceSettings,
  useUpdateTraceSettings,
} from "@/shared/api/traces"
import { docsSourceHref } from "@/shared/helpers/docs"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"

const LEVELS: { value: ContentCaptureLevel; label: string }[] = [
  { value: "off", label: "Off" },
  { value: "tool_io", label: "Tool calls" },
  { value: "full", label: "Everything" },
]

const ORDER: ContentCaptureLevel[] = ["off", "tool_io", "full"]

const READERS =
  "It is kept encrypted and is readable by the person who ran the session; by organization admins only if you allow it below; and by platform operators only through a recorded break-glass with a stated reason. It is deleted when the deployment's content retention ends (7 days unless the operator changed it)."

const READS_PAGE_SIZE = 25

// What letting organization admins read content means, said before an admin
// commits to it either way.
function adminAccessConfirmation(allow: boolean, workspace: string) {
  if (allow) {
    return {
      heading: `Let organization admins read content in ${workspace}?`,
      body: "The organization's owners and admins will be able to read the captured prompts, outputs and tool results of every session in this workspace, not only their own. Every read they make is recorded with who made it and when, and listed under Content reads on this page.",
      confirmLabel: "Let admins read",
    }
  }
  return {
    heading: `Stop organization admins reading content in ${workspace}?`,
    body: "From then on only the person who ran a session can read its content. Reads already made stay recorded under Content reads.",
    confirmLabel: "Stop admin reads",
  }
}

// What each level means, said before an admin commits to it.
function confirmation(level: ContentCaptureLevel, workspace: string) {
  if (level === "off") {
    return {
      heading: `Stop capturing content in ${workspace}?`,
      body: "Takes effect within 30 seconds: requests after that keep no prompt, output or tool content. What is already stored stays readable until its retention ends, unless you purge it.",
      confirmLabel: "Turn off",
    }
  }
  if (level === "tool_io") {
    return {
      heading: `Capture tool calls in ${workspace}?`,
      body: `From now on, each tool call's arguments and result are stored with its session. ${READERS}`,
      confirmLabel: "Capture tool calls",
    }
  }
  return {
    heading: `Capture everything in ${workspace}?`,
    body: `From now on, each request's input, the model's output for it, and every tool call's arguments and result are stored with their session. ${READERS}`,
    confirmLabel: "Capture everything",
  }
}

/**
 * The controls for what a workspace's traces keep beyond their metadata and who
 * may read it, behind a confirmation each: the level, up to the deployment's
 * ceiling, whether organization admins may read it, and a purge of what was kept.
 */
export function TraceContentControls({
  workspace,
  settings,
  onChange,
  onAdminAccessChange,
  onPurge,
  isSaving,
  isSavingAdminAccess,
  isPurging,
  saveError,
  adminAccessError,
  purgeError,
}: {
  workspace: string
  settings: TraceSettings
  /** Called once the admin confirms; `done` closes the dialog on success. */
  onChange: (level: ContentCaptureLevel, done: () => void) => void
  /** Called once the admin confirms; `done` closes the dialog on success. */
  onAdminAccessChange: (allow: boolean, done: () => void) => void
  onPurge: (done: () => void) => void
  isSaving: boolean
  isSavingAdminAccess: boolean
  isPurging: boolean
  /** The server's refusal of a level change, shown in its confirmation. */
  saveError: unknown
  adminAccessError: unknown
  purgeError: unknown
}) {
  const [pending, setPending] = useState<ContentCaptureLevel | null>(null)
  const [pendingAdminAccess, setPendingAdminAccess] = useState<
    boolean | undefined
  >(undefined)
  const [purging, setPurging] = useState(false)
  const adminConfirm =
    pendingAdminAccess === undefined
      ? null
      : adminAccessConfirmation(pendingAdminAccess, workspace)
  const permitted = LEVELS.filter(
    (level) => ORDER.indexOf(level.value) <= ORDER.indexOf(settings.ceiling),
  )
  const confirm = pending ? confirmation(pending, workspace) : null
  return (
    <>
      <SettingsGroup isBounded>
        <SettingRow
          label="Content capture"
          help={
            settings.ceiling === "off"
              ? "This deployment does not permit content capture. Its operator sets the most any workspace may keep, and can permit it once content encryption is configured."
              : settings.content_capture === "off"
                ? "Sessions keep timing, tokens and cost only. No prompt, output or tool content is stored."
                : READERS
          }
          control={
            <Segmented
              label={`Content capture for ${workspace}`}
              options={permitted}
              value={settings.content_capture}
              onChange={(next) => {
                if (next !== settings.content_capture) {
                  setPending(next as ContentCaptureLevel)
                }
              }}
              size="sm"
            />
          }
        />
        <SettingRow
          label="Let organization admins read content"
          help={
            settings.admin_content_access
              ? "The organization's owners and admins can read every session's content in this workspace. Each read is recorded below."
              : "Only the person who ran a session can read its content."
          }
          control={
            <Toggle
              label={`Let organization admins read content in ${workspace}`}
              isSelected={settings.admin_content_access}
              onChange={setPendingAdminAccess}
            />
          }
        />
        <SettingRow
          label="Stored content"
          help="Delete every prompt, output and tool result captured in this workspace, and destroy each session's key. Sessions and their timing, tokens and cost stay."
          control={
            <Button variant="danger" onPress={() => setPurging(true)}>
              Purge stored content
            </Button>
          }
        />
      </SettingsGroup>
      {confirm && pending ? (
        <ConfirmDialog
          isOpen
          onOpenChange={(open) => {
            if (!open) setPending(null)
          }}
          heading={confirm.heading}
          body={confirm.body}
          confirmLabel={confirm.confirmLabel}
          confirmVariant="primary"
          isPending={isSaving}
          error={saveError}
          onConfirm={() => onChange(pending, () => setPending(null))}
        />
      ) : null}
      {adminConfirm && pendingAdminAccess !== undefined ? (
        <ConfirmDialog
          isOpen
          onOpenChange={(open) => {
            if (!open) setPendingAdminAccess(undefined)
          }}
          heading={adminConfirm.heading}
          body={adminConfirm.body}
          confirmLabel={adminConfirm.confirmLabel}
          confirmVariant="primary"
          isPending={isSavingAdminAccess}
          error={adminAccessError}
          onConfirm={() =>
            onAdminAccessChange(pendingAdminAccess, () =>
              setPendingAdminAccess(undefined),
            )
          }
        />
      ) : null}
      <ConfirmDialog
        isOpen={purging}
        onOpenChange={setPurging}
        heading={`Purge stored content in ${workspace}?`}
        body="Every captured prompt, output and tool result in this workspace is deleted from the database, and each session's key is destroyed. Copies in database backups remain until those backups expire. Sessions stay. This cannot be undone."
        confirmLabel="Purge content"
        confirmVariant="danger"
        isPending={isPurging}
        error={purgeError}
        onConfirm={() => onPurge(() => setPurging(false))}
      />
    </>
  )
}

// The workspace's recorded content reads, a page at a time.
function ContentReadsSection({ workspaceId }: { workspaceId: string }) {
  const hosts = useSurfaces()
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(READS_PAGE_SIZE)
  const reads = useContentAccessLog(workspaceId, page, pageSize)
  // Best effort: a reader the roster does not name keeps a short id.
  const members = useOrganizationMembers(hosts("organizations"))
  const names = new Map(
    (members.data ?? []).flatMap((member): [string, string][] =>
      member.user_id ? [[member.user_id, memberLabel(member)]] : [],
    ),
  )
  const items = reads.data?.items ?? []
  return (
    <section className="flex flex-col gap-2" aria-labelledby="content-reads">
      <h2 id="content-reads" className="text-title">
        Content reads
      </h2>
      <p className="text-body text-muted">
        Every read of this workspace's captured content: by the person who ran
        the session, by an organization admin, or by a platform operator
        breaking glass with a stated reason.
      </p>
      {reads.isError && !reads.data ? (
        <ErrorBanner error={reads.error} />
      ) : (
        <>
          <ContentReadsList
            reads={items}
            names={names}
            isLoading={reads.isPending && !reads.data}
          />
          {page > 0 || items.length === pageSize ? (
            <TablePagination
              page={page}
              pageSize={pageSize}
              total={null}
              rowsOnPage={items.length}
              onPageChange={setPage}
              onPageSizeChange={(size) => {
                setPageSize(size)
                setPage(0)
              }}
              isFetching={reads.isFetching}
              hasNextFallback={items.length === pageSize}
              label="content reads"
            />
          ) : null}
        </>
      )}
    </section>
  )
}

/** Tools → Trace content: what the selected workspace's sessions keep. */
export function TraceContentPage() {
  const hosts = useSurfaces()
  const { selected } = useSelectedWorkspace()
  const context = useOrganizationContext()
  // Only a workspace's owners and admins may read its setting, so a member is
  // told so rather than shown the server's refusal.
  const manages = canManageWorkspace(context.data, selected?.role)
  const workspaceId =
    hosts("traces") && manages ? (selected?.workspace_id ?? "") : ""
  const settings = useTraceSettings(workspaceId)
  const setCapture = useUpdateTraceSettings(workspaceId)
  const setAdminAccess = useUpdateTraceSettings(workspaceId)
  const purge = usePurgeTraceContent(workspaceId)

  let body: ReactNode
  if (!hosts("traces")) {
    body = (
      <EmptyMessage>
        This deployment does not record agent sessions.
      </EmptyMessage>
    )
  } else if (!manages) {
    body = (
      <EmptyMessage>
        Only owners and admins of this workspace can change what its sessions
        keep.
      </EmptyMessage>
    )
  } else if (settings.data && selected) {
    body = (
      <>
        <TraceContentControls
          workspace={selected.name}
          settings={settings.data}
          onChange={(level, done) =>
            setCapture.mutate({ content_capture: level }, { onSuccess: done })
          }
          onAdminAccessChange={(allow, done) =>
            setAdminAccess.mutate(
              { admin_content_access: allow },
              { onSuccess: done },
            )
          }
          onPurge={(done) => purge.mutate(undefined, { onSuccess: done })}
          isSaving={setCapture.isPending}
          isSavingAdminAccess={setAdminAccess.isPending}
          isPurging={purge.isPending}
          saveError={setCapture.error}
          adminAccessError={setAdminAccess.error}
          purgeError={purge.error}
        />
        <ContentReadsSection key={workspaceId} workspaceId={workspaceId} />
      </>
    )
  } else {
    body = <ErrorBanner error={settings.error} />
  }

  return (
    <div className="flex flex-col gap-6">
      <PageIntro title="Trace content" docsHref={docsSourceHref("traces.md")}>
        <p className="text-body text-muted">
          Every agent session records its steps, timing, tokens and cost. What
          the steps said (prompts, outputs, tool arguments and results) is kept
          only when a workspace admin turns it on here, and only on a deployment
          whose operator configured content encryption. It is readable by the
          person who ran the session; by organization admins only if you allow
          it below; and by platform operators only through a recorded
          break-glass with a stated reason.
        </p>
      </PageIntro>
      {body}
    </div>
  )
}
