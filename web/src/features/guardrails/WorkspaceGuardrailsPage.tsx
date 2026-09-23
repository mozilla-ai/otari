import { Link } from "@tanstack/react-router"

import type { OrganizationGuardrail } from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { mandateConsequence } from "@/features/guardrails/buildState"
import { MandateConsequenceMark } from "@/features/guardrails/GuardrailStatus"
import {
  ifItCantRunLabel,
  mandatesFor,
  runsOnLabel,
} from "@/features/guardrails/guardrailRows"
import { canManage } from "@/features/organization/roles"
import { useOrganizationGuardrailDefinitions } from "@/shared/api/guardrails"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useOrganizationGuardrails } from "@/shared/api/tools"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// Which of the organization's guardrails run on this workspace's requests, and
// nothing else: read-only, with changes made on the organization page the
// intro links to. A paused mandate runs nowhere, so it is not listed.
//
// The definitions are read but not shown: they name what a mandate runs on and
// whether its guardrail is running.
//
// Owners and admins only, for the reason the organization page is: the server
// refuses anyone else both lists, since they name the vendors the organization
// has accounts with.

export function WorkspaceGuardrailsPage() {
  const context = useOrganizationContext()
  const canRead = canManage(context.data)
  const { selected } = useSelectedWorkspace()
  const definitions = useOrganizationGuardrailDefinitions(canRead)
  const mandates = useOrganizationGuardrails(canRead)

  const defined = definitions.data ?? []
  const running = selected
    ? mandatesFor(selected.workspace_id, mandates.data ?? []).filter(
        (mandate) => mandate.enabled,
      )
    : []

  const columns: DataTableColumn<OrganizationGuardrail>[] = [
    {
      id: "profile",
      header: "Profile",
      isRowHeader: true,
      cell: (row) => (
        <div className="flex flex-col gap-1">
          <code className="font-mono text-foreground">{row.profile}</code>
          <MandateConsequenceMark
            consequence={mandateConsequence(row, defined)}
          />
        </div>
      ),
    },
    {
      id: "runs_on",
      header: "Runs on",
      cell: (row) => (
        <span className="text-muted">{runsOnLabel(row, defined)}</span>
      ),
    },
    {
      id: "mode",
      header: "Mode",
      cell: (row) => (
        <span className="text-muted">
          {row.mode === "block" ? "Block" : "Monitor"}
        </span>
      ),
    },
    {
      id: "on_unavailable",
      header: "If it can't run",
      cell: (row) => (
        <span className="text-muted">{ifItCantRunLabel(row)}</span>
      ),
    },
  ]

  return (
    <div className="flex flex-col">
      <PageIntro title="Guardrails">
        The guardrails your organization runs on what users in{" "}
        {selected?.name ?? "this workspace"} send to a model. An owner or admin
        changes them on the organization&rsquo;s{" "}
        <Link
          to="/organization/guardrails"
          className="font-medium text-link hover:text-link-hover"
        >
          Guardrails
        </Link>{" "}
        page.
      </PageIntro>

      <ErrorBanner
        error={context.error ?? definitions.error ?? mandates.error}
      />

      {context.data && !canRead ? (
        <InfoBanner>
          Guardrails are set by an owner or admin of the organization.
        </InfoBanner>
      ) : null}

      {canRead || context.isPending ? (
        <TableScrollFrame className="otari-guardrail-mandates-table">
          <DataTable
            ariaLabel="Guardrails running in this workspace"
            columns={columns}
            rows={running}
            getRowKey={(row) => row.id}
            isLoading={context.isPending || mandates.isLoading}
            emptyContent="No guardrail runs here, so only the guardrails a caller asks for run."
          />
        </TableScrollFrame>
      ) : null}
    </div>
  )
}
