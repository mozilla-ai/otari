import { useState } from "react"
import { FiEdit2, FiTrash2 } from "react-icons/fi"

import type { AgentRecommender, ConfiguredRecommender } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useAgentRecommenders,
  useDeleteAgentRecommender,
  useUpdateAgentRecommender,
} from "@/shared/api/agentRecommenders"
import { ApiError } from "@/shared/api/client"
import { useDeploymentAdminAccess } from "@/shared/api/deployment"

import { AgentRecommenderForm } from "./AgentRecommenderForm"
import { priceSummary, providerLabel } from "./recommenderDraft"

// The Subagent router: the decision models that pick a subagent's model, and
// what one recommendation costs. When a coding agent using Otari starts a
// subagent, the router switched on here picks the model it runs on, and the
// caller is charged its tokens at the rates set here. The operator may add
// several; one is on at a time.
//
// Operator-only, and the API refuses anyone else with a 404, so the page asks
// the deployment admin gate first, as Hosted providers does, and words a
// refusal as the page not being theirs rather than as an error.

const TITLE = "Subagent router"

function configuredFallback(configured: ConfiguredRecommender | null): string {
  return configured
    ? `Until one is added, ${configured.selector} from config.yml answers, unpriced.`
    : "Until one is added, the decision model agent_recommender_model names in config.yml answers, unpriced, if it is set."
}

export function AgentRecommendersPage() {
  const access = useDeploymentAdminAccess()
  // Withheld until the gate answers, so a non-operator load does not put a 404
  // in the console to learn what the gate is about to say.
  const isGranted = access.data === true
  const recommenders = useAgentRecommenders(isGranted)
  const update = useUpdateAgentRecommender()
  const remove = useDeleteAgentRecommender()

  const [isAdding, setIsAdding] = useState(false)
  const [addOpenCount, setAddOpenCount] = useState(0)
  const [editingId, setEditingId] = useState("")
  const [pendingRemoveId, setPendingRemoveId] = useState("")

  const openAdd = () => {
    setEditingId("")
    setAddOpenCount((n) => n + 1)
    setIsAdding(true)
  }

  if (access.isLoading) {
    return <PageLoading label="Loading the subagent routers…" />
  }

  // A gate that failed to answer is not a gate that said no.
  if (access.isError) {
    return (
      <div className="flex flex-col">
        <PageIntro title={TITLE} />
        <ErrorBanner error={access.error} />
      </div>
    )
  }

  // The list's 404 is the router's operator refusal, the same predicate as the
  // gate's, so a disagreement between the two still offers no write.
  const isListRefused =
    recommenders.error instanceof ApiError && recommenders.error.status === 404
  if (!isGranted || isListRefused) {
    return (
      <div className="flex flex-col">
        <PageIntro title={TITLE} />
        <EmptyState
          title="Subagent router is not available to you"
          description="Choosing the model that picks a subagent's model is for this deployment's operators. Ask one of them if you need access here."
        />
      </div>
    )
  }

  const rows = recommenders.data?.data ?? []
  const configured = recommenders.data?.configured ?? null
  const editing = rows.find((row) => row.id === editingId)
  const pendingRemove = rows.find((row) => row.id === pendingRemoveId)
  const isEmpty =
    !recommenders.error && !recommenders.isPending && rows.length === 0
  const noneOn = rows.length > 0 && !rows.some((row) => row.enabled)

  const columns: DataTableColumn<AgentRecommender>[] = [
    {
      id: "model",
      header: "Model",
      isRowHeader: true,
      cell: (row) => row.model,
    },
    {
      id: "provider",
      header: "Provider",
      cell: (row) => providerLabel(row.provider),
    },
    {
      id: "api_base",
      header: "Endpoint",
      cell: (row) =>
        row.api_base ?? <span className="text-muted">Provider default</span>,
    },
    {
      id: "key",
      header: "API key",
      cell: (row) =>
        !row.has_api_key ? (
          <Badge tone="warn">No key stored</Badge>
        ) : row.api_key_last4 ? (
          <span className="font-mono text-caption">
            ••••{row.api_key_last4}
          </span>
        ) : (
          <span className="text-caption">Stored</span>
        ),
    },
    {
      id: "price",
      header: "Price",
      cell: (row) => priceSummary(row),
    },
    {
      id: "enabled",
      header: "Serving",
      cell: (row) => (
        <Toggle
          isSelected={row.enabled}
          isDisabled={update.isPending && update.variables?.id === row.id}
          label={`Route with ${row.model}`}
          onChange={(enabled) =>
            update.mutate({ id: row.id, body: { enabled } })
          }
        />
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiEdit2}
            label="Edit"
            ariaLabel={`Edit ${row.model}`}
            onPress={() => {
              setIsAdding(false)
              setEditingId(row.id)
            }}
          />
          <RowAction
            icon={FiTrash2}
            label="Remove"
            ariaLabel={`Remove ${row.model}`}
            onPress={() => setPendingRemoveId(row.id)}
          />
        </RowActionRow>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title={TITLE}
        action={
          <Button variant="primary" onPress={openAdd}>
            Add router
          </Button>
        }
      >
        When a coding agent using Otari starts a subagent, the router switched
        on here picks the model it runs on. Each recommendation is billed to the
        caller: its tokens at the rates set here.
      </PageIntro>
      {noneOn ? (
        <InfoBanner>
          No router is switched on, so no model is picked, and the decision
          model config.yml names is not asked either. The plugin starts each
          subagent on its usual model.
        </InfoBanner>
      ) : null}
      <ErrorBanner error={recommenders.error ?? update.error} />
      <AgentRecommenderForm
        key={addOpenCount}
        startFrom={rows.length === 0 && configured ? configured : undefined}
        isOpen={isAdding}
        onClose={() => setIsAdding(false)}
      />
      {editing ? (
        // Remounted per row: the draft is seeded from the row once.
        <AgentRecommenderForm
          key={editing.id}
          recommender={editing}
          isOpen
          onClose={() => setEditingId("")}
        />
      ) : null}
      {isEmpty ? (
        <EmptyState
          title="No router added"
          description={`${configuredFallback(configured)} Otherwise the plugin starts each subagent on its usual model.`}
          actionLabel="Add router"
          onAction={openAdd}
        />
      ) : (
        <TableScrollFrame className="otari-subagent-routers-table">
          <DataTable
            ariaLabel="Subagent routers"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={recommenders.isLoading}
          />
        </TableScrollFrame>
      )}
      <ConfirmDialog
        isOpen={pendingRemove !== undefined}
        onOpenChange={(open) => {
          if (!open) setPendingRemoveId("")
        }}
        heading="Remove this router?"
        body={
          pendingRemove
            ? `${pendingRemove.model} and its stored key are deleted. With no router left, the decision model config.yml names answers, unpriced, or nothing does if it is not set.`
            : ""
        }
        confirmLabel="Remove router"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingRemove) return
          remove.mutate(pendingRemove.id, {
            onSuccess: () => setPendingRemoveId(""),
          })
        }}
      />
    </div>
  )
}
