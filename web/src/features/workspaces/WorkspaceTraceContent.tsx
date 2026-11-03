import { useState } from "react"

import type { Workspace } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Toggle } from "@/design-system/forms/Toggle"
import { exceptionalAccess } from "@/features/traces/exceptionalAccess"
import { useTraceContentPolicy } from "@/shared/api/traces"
import {
  useDeleteWorkspaceTraceContent,
  useSetWorkspaceTraceContent,
} from "@/shared/api/workspaces"

const LABEL = "Keep what agent sessions were about"

type Pending = "on" | "off" | "delete"

/**
 * Whether a workspace keeps what its agent sessions were about: the prompts,
 * commands and file paths its coding agents report.
 *
 * The explanation is always on the page rather than in a tooltip, because a
 * tooltip does not exist on a phone and this is what a manager has to read
 * before they act. It says outright who can read the content and when it goes,
 * because "in the same workspace" is what people will otherwise assume. Every
 * change asks first. Deleting what is stored is offered on its own as well as
 * with turning content off, so it stays reachable while content is off.
 */
export function WorkspaceTraceContent({ workspace }: { workspace: Workspace }) {
  const { policy, isPending: policyPending } = useTraceContentPolicy()
  const available = policy.content_available
  const setContent = useSetWorkspaceTraceContent()
  const deleteContent = useDeleteWorkspaceTraceContent()
  const [pending, setPending] = useState<Pending>()
  const [deleteStored, setDeleteStored] = useState(false)
  const enabled = workspace.trace_content_enabled ?? false
  const exceptions = exceptionalAccess(policy.platform_break_glass)
  const retention = `A session is deleted, with what it was about, ${policy.retention_days} days after its last activity, and ${policy.session_max_age_days} days after it started at the latest.`

  const close = () => {
    setPending(undefined)
    setDeleteStored(false)
    // Cleared on the way out, so an earlier action's error never shows on the next dialog.
    setContent.reset()
    deleteContent.reset()
  }
  const confirm = async () => {
    // A failure stays on the dialog, through `error` below, rather than escaping as an unhandled rejection.
    try {
      if (pending === "on" || pending === "off") {
        await setContent.mutateAsync({
          id: workspace.id,
          enabled: pending === "on",
        })
      }
      if (pending === "delete" || (pending === "off" && deleteStored)) {
        await deleteContent.mutateAsync(workspace.id)
      }
      close()
    } catch {
      // Shown by the dialog.
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-body">{LABEL}</p>
          <p className="mt-1 max-w-md text-caption">
            {available
              ? `Stores the prompt that starts each turn, the commands and file paths your agents' tools use, and what guardrail checks looked at. Never tool output or model responses. It is stored encrypted, though not end to end: this deployment holds the keys, so it can read it. Otari shows it only to the person whose session it is, unless they share a session for review. ${exceptions} Other members of this workspace, including its admins, see that a session ran and what it cost, but not what it was about. It is collected only for the workspace's current members; what is already stored stays until it is deleted, after its owner leaves too. Your agents also have to be set to send it. ${retention}`
              : `This deployment does not keep session content, so a workspace cannot turn it on. Sessions still record their tools, costs and outcomes. ${retention}`}
          </p>
        </div>
        <Toggle
          isSelected={available && enabled}
          onChange={(next) => setPending(next ? "on" : "off")}
          label={LABEL}
          isDisabled={!available || policyPending || setContent.isPending}
        />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="ghost"
          onPress={() => setPending("delete")}
          isDisabled={deleteContent.isPending}
        >
          Delete stored session content
        </Button>
      </div>
      <ErrorBanner error={setContent.error ?? deleteContent.error} />
      <ConfirmDialog
        isOpen={pending !== undefined}
        onOpenChange={(open) => {
          if (!open) close()
        }}
        heading={
          pending === "on"
            ? "Keep what this workspace's sessions were about?"
            : pending === "off"
              ? "Stop keeping what sessions were about?"
              : "Delete what this workspace's sessions were about?"
        }
        body={
          pending === "on" ? (
            <p>
              From the next export on, this workspace keeps the prompts,
              commands and file paths its members' agents send. Otari shows each
              person only their own sessions' content. Nobody else sees it, you
              included, unless they share a session with you. {exceptions}{" "}
              {retention}
            </p>
          ) : pending === "off" ? (
            <div className="flex flex-col gap-3">
              <p>
                Nothing new is kept from the next export on, including content
                already sent and not yet stored. What is already stored stays
                until its sessions are deleted, unless you delete it now.
              </p>
              <Checkbox
                isSelected={deleteStored}
                onChange={setDeleteStored}
                hasTouchTarget
              >
                Also delete the content already stored
              </Checkbox>
            </div>
          ) : (
            <p>
              Deletes the prompts, commands and file paths stored for every
              session in this workspace, and anything sent but not yet stored.
              The sessions, their costs and outcomes stay. This cannot be
              undone.
            </p>
          )
        }
        confirmLabel={
          pending === "on"
            ? "Keep content"
            : pending === "off"
              ? "Stop keeping content"
              : "Delete content"
        }
        confirmVariant={
          pending === "delete" || (pending === "off" && deleteStored)
            ? "danger"
            : "primary"
        }
        isPending={setContent.isPending || deleteContent.isPending}
        error={setContent.error ?? deleteContent.error}
        onConfirm={() => void confirm()}
      />
    </div>
  )
}
