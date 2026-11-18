import { useState } from "react"
import { useIsOrganizationManager } from "@/features/organization/access"
import { attributionId } from "@/features/organization/attribution"
import { memberRowKey } from "@/features/organization/roles"
import { type PersonRow, personRows } from "@/features/people/people"
import { type SpendWindow, windowOf } from "@/features/spend/period"
import { readState, type Spend, useSpend } from "@/features/spend/useSpend"
import { useKeys, useKeysScope } from "@/shared/api/apiKeys"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import { useSessionCountSince, useSessionsSince } from "@/shared/api/traces"
import { useRefusalsSince } from "@/shared/api/usage"
import { useWorkspaces } from "@/shared/api/workspaces"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * "This month" on People, as Home reads it: the calendar month so far, over
 * one clock for the page's life so the queries keyed on it hold still.
 */
export function useMonth(): { window: SpendWindow; now: number } {
  const [now] = useState(() => Date.now())
  return { window: windowOf("month", now), now }
}

interface People {
  rows: PersonRow[]
  /** False when the month held more sessions than were read. */
  isSessionCountComplete: boolean
  /**
   * Every spend ceiling, with ended periods read as nothing spent, for a
   * manager; undefined for anyone the read refuses.
   */
  ceilings: Spend["ceilings"] | undefined
  /**
   * The workspace every per-person figure is read for: the rail's selected one
   * on the list, or undefined for the whole organization.
   */
  workspaceId: string | undefined
  isLoading: boolean
  error: unknown
}

/**
 * The roster for a month. `inSelectedWorkspace` lists only the people of the
 * workspace the rail has selected, and reads every figure for it (the whole
 * organization for a caller with none selected). Without it, everyone is
 * listed and every figure is the caller's whole view across workspaces, so a
 * person's own page opens for someone outside the selected workspace and
 * shows what they did there. `personId` is a person's page: only that
 * person's sessions are counted, in one read.
 */
export function usePeople(
  window: SpendWindow,
  now: number,
  {
    inSelectedWorkspace = false,
    personId,
  }: { inSelectedWorkspace?: boolean; personId?: string } = {},
): People {
  const { selected, isSettled } = useSelectedWorkspace()
  const isManager = useIsOrganizationManager()
  const contextRead = useOrganizationContext()
  const context = contextRead.data
  const members = useOrganizationMembers()
  const workspaces = useWorkspaces()
  // Home's own read, for the whole organization: the same rows, the same
  // query, so a person's spend and tokens here are what Home's People cut says.
  // No "vs previous" on People or a person's page, so no previous window.
  // On the list every figure is the selected workspace's, as the roster is;
  // with none selected, and on a person's page, it is every workspace the
  // caller reads.
  const isReady = !inSelectedWorkspace || isSettled
  // A deployment that records no sessions serves no route to read them from.
  const hasSessions = useSurfaces()("traces")
  const workspaceId = inSelectedWorkspace ? selected?.workspace_id : undefined
  const spend = useSpend(window, {
    workspaceId,
    isEnabled: isReady,
    resetEndedAt: now,
    withPrevious: false,
  })
  const keysScope = useKeysScope()
  // `useSpend` reads the key list too; this is the same query, asked once.
  const keys = useKeys()
  const refusals = useRefusalsSince(window.start, workspaceId, isReady)
  const personUserId = personId
    ? (members.data ?? [])
        .filter((member) => memberRowKey(member) === personId)
        .map(attributionId)
        .find((id) => id !== null)
    : undefined
  const sessions = useSessionsSince(
    window.start,
    workspaceId,
    isReady && hasSessions && personId === undefined,
  )
  const personSessions = useSessionCountSince(
    window.start,
    personUserId,
    isReady && hasSessions && personId !== undefined,
  )
  const sessionRead = personId === undefined ? sessions : personSessions
  const ownMembership = context?.organization_member_id
  return {
    rows: personRows(members.data ?? [], {
      spendRows: spend.people,
      refusalsByUser: refusals.data?.by_user ?? [],
      sessions: sessions.data?.sessions ?? [],
      sessionCounts:
        personUserId && personSessions.data
          ? new Map([[personUserId, personSessions.data.count]])
          : undefined,
      unread: {
        spend: spend.reads.usage === "failed",
        sessions: readState(sessionRead) === "failed",
        refusals: readState(refusals) === "failed",
        keys: readState(keys) === "failed",
      },
      keys: keys.data ?? [],
      now,
      keyReaderMemberIds: keysScope.isDeploymentWide
        ? undefined
        : new Set(ownMembership ? [ownMembership] : []),
      workspaceId: inSelectedWorkspace ? selected?.workspace_id : undefined,
      workspaceIds: workspaces.data
        ? new Set(workspaces.data.map((one) => one.id))
        : undefined,
      visibleWorkspaceIds: isManager
        ? undefined
        : new Set(
            (context?.workspace_memberships ?? []).map(
              (one) => one.workspace_id,
            ),
          ),
    }),
    isSessionCountComplete: sessions.data?.isComplete ?? true,
    ceilings: isManager ? spend.ceilings : undefined,
    workspaceId,
    isLoading:
      !isReady ||
      spend.isLoading ||
      keys.isLoading ||
      workspaces.isLoading ||
      refusals.isLoading ||
      sessionRead.isLoading,
    error:
      spend.error ??
      members.error ??
      keys.error ??
      workspaces.error ??
      refusals.error ??
      sessionRead.error ??
      contextRead.error,
  }
}
