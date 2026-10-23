import { useState } from "react"

import type { Workspace } from "@/client"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Toggle } from "@/design-system/forms/Toggle"
import {
  useDeleteWorkspaceTraceContent,
  useSetWorkspaceTraceContent,
} from "@/shared/api/workspaces"
import { useDeployment } from "@/shared/hooks/useDeployment"

const LABEL = "Keep what agent sessions were about"

/**
 * Whether a workspace keeps what its agent sessions were about: the prompts,
 * commands and file paths its coding agents report.
 *
 * The explanation is always on the page rather than in a tooltip, because a
 * tooltip does not exist on a phone and this is what a manager has to read
 * before they act. It says outright who can read the content, because "in the
 * same workspace" is what people will otherwise assume. Either direction asks
 * first: turning it on starts keeping prompts, and turning it off can also
 * delete what was kept.
 */
export function WorkspaceTraceContent({ workspace }: { workspace: Workspace }) {
  const available = useDeployment().trace_content_available
  const setContent = useSetWorkspaceTraceContent()
  const deleteContent = useDeleteWorkspaceTraceContent()
  const [pending, setPending] = useState<"on" | "off">()
  const [deleteStored, setDeleteStored] = useState(false)
  const enabled = workspace.trace_content_enabled ?? false

  const close = () => {
    setPending(undefined)
    setDeleteStored(false)
  }
  const confirm = async () => {
    await setContent.mutateAsync({
      id: workspace.id,
      enabled: pending === "on",
    })
    if (pending === "off" && deleteStored) {
      await deleteContent.mutateAsync(workspace.id)
    }
    close()
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-body">{LABEL}</p>
          <p className="mt-1 max-w-md text-caption">
            {available
              ? "Stores the prompt that starts each turn, the commands and file paths your agents' tools use, and what guardrail checks looked at. Never tool output or model responses. It is encrypted, and only the person whose session it is can read it, unless they share a session for review. Other members of this workspace, including its admins, see that a session ran and what it cost, but not what it was about. Your agents also have to be set to send it."
              : "This deployment does not keep session content, so a workspace cannot turn it on. Sessions still record their tools, costs and outcomes."}
          </p>
        </div>
        <Toggle
          isSelected={available && enabled}
          onChange={(next) => setPending(next ? "on" : "off")}
          label={LABEL}
          isDisabled={!available || setContent.isPending}
        />
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
            : "Stop keeping what sessions were about?"
        }
        body={
          pending === "on" ? (
            <p>
              From the next export on, this workspace keeps the prompts,
              commands and file paths its members' agents send. Each person
              reads only their own sessions' content. Nobody else does, you
              included, unless they share a session with you.
            </p>
          ) : (
            <div className="flex flex-col gap-3">
              <p>
                Nothing new is kept from the next export on. What is already
                stored stays until its sessions age out, unless you delete it
                now.
              </p>
              <Checkbox
                isSelected={deleteStored}
                onChange={setDeleteStored}
                hasTouchTarget
              >
                Also delete the content already stored
              </Checkbox>
            </div>
          )
        }
        confirmLabel={
          pending === "on" ? "Keep content" : "Stop keeping content"
        }
        confirmVariant={
          pending === "off" && deleteStored ? "danger" : "primary"
        }
        isPending={setContent.isPending || deleteContent.isPending}
        error={setContent.error ?? deleteContent.error}
        onConfirm={() => void confirm()}
      />
    </div>
  )
}
