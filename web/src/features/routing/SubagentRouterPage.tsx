import { useState } from "react"
import { FiEdit2, FiTrash2 } from "react-icons/fi"

import type { AgentRecommender } from "@/client"
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
import { useRecommenderMargin } from "@/features/routing/overlayRecommenderMargin"
import {
  useAgentRecommenders,
  useRemoveAgentRecommender,
  useUpdateAgentRecommender,
} from "@/shared/api/agentRecommenders"
import { ApiError } from "@/shared/api/client"
import { useDeploymentAdminAccess } from "@/shared/api/deployment"
import { UnavailableHere } from "@/shared/components/access/UnavailableHere"
import { useSurfaces } from "@/shared/hooks/useDeployment"

import { priceSummary, providerLabel } from "./recommenderDraft"
import { SubagentRouterForm } from "./SubagentRouterForm"

// The decision models that pick the model a coding agent's subagent starts on,
// and what one recommendation costs. When someone using the otari-router
// Claude Code plugin starts a subagent, the router switched on here answers,
// and the call is billed to them at its rates. The operator may add several;
// one is on at a time. While none is added, `agent_recommender_model` in
// config.yml answers.
//
// Operator-only, and the API refuses anyone else with a 404, so the page asks
// the deployment admin gate first, as Hosted providers does, and words a
// refusal as the page not being theirs rather than as an error.

export function SubagentRouterPage() {
  const hostsSurface = useSurfaces()
  // Before any hook that asks the API, so an unpublished surface asks nothing.
  if (!hostsSurface("subagent_router")) {
    return <UnavailableHere title="Subagent router" />
  }
  return <SubagentRouterContent />
}

function SubagentRouterContent() {
  const access = useDeploymentAdminAccess()
  // Withheld until the gate answers, so a non-operator load does not put a 404
  // in the console to learn what the gate is about to say.
  const isGranted = access.data === true
  const recommenders = useAgentRecommenders(isGranted)
  const update = useUpdateAgentRecommender()
  const remove = useRemoveAgentRecommender()
  const margin = useRecommenderMargin()

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
        <PageIntro title="Subagent router" />
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
        <PageIntro title="Subagent router" />
        <EmptyState
          title="Subagent router is not available to you"
          description="Choosing the decision model that picks a subagent's model is for this deployment's operators. Ask one of them if you need access here."
        />
      </div>
    )
  }

  const rows = recommenders.data?.data ?? []
  const providers = recommenders.data?.decision_providers ?? []
  const editing = rows.find((row) => row.id === editingId)
  const pendingRemove = rows.find((row) => row.id === pendingRemoveId)
  // A failed list is not an empty one.
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
      cell: (row) => (
        <span className="text-muted">{row.api_base ?? "provider default"}</span>
      ),
    },
    {
      id: "key",
      header: "API key",
      cell: (row) =>
        !row.has_api_key ? (
          <Badge tone="warn">No key stored</Badge>
        ) : row.api_key_last4 ? (
          <code className="text-muted">••••{row.api_key_last4}</code>
        ) : (
          <code className="text-muted">••••</code>
        ),
    },
    {
      id: "price",
      header: "Price",
      cell: (row) => `${priceSummary(row)}${margin.summary(row.id)}`,
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
            update.mutate({ id: row.id, patch: { enabled } })
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
    <div className="flex flex-col">
      <PageIntro
        title="Subagent router"
        action={
          <Button
            variant="primary"
            className="min-h-11 md:min-h-9"
            onPress={openAdd}
          >
            Add router
          </Button>
        }
      >
        When someone using the otari-router plugin for Claude Code starts a
        subagent, the router switched on here picks the model it runs on. Each
        recommendation is billed to them: the decision model's tokens, at the
        rates set here.
      </PageIntro>

      {noneOn ? (
        <InfoBanner>
          No router is switched on, so no model is picked, even with
          agent_recommender_model set in config.yml. The plugin starts each
          subagent on its usual model.
        </InfoBanner>
      ) : null}
      <ErrorBanner error={recommenders.error ?? update.error} />

      {isEmpty ? (
        <EmptyState
          title="No router added"
          description="Until one is added, the decision model agent_recommender_model names in config.yml answers, if its provider is configured. A router added here replaces it."
          actionLabel="Add router"
          onAction={openAdd}
        />
      ) : (
        <DataTable
          ariaLabel="Subagent routers"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.id}
          isLoading={recommenders.isLoading}
        />
      )}

      <SubagentRouterForm
        key={addOpenCount}
        providers={providers}
        switchOn={!rows.some((row) => row.enabled)}
        isOpen={isAdding}
        onClose={() => setIsAdding(false)}
      />
      {editing ? (
        // Remounted per row: the draft is seeded from the row once.
        <SubagentRouterForm
          key={editing.id}
          recommender={editing}
          providers={providers}
          switchOn={editing.enabled}
          isOpen
          onClose={() => setEditingId("")}
        />
      ) : null}

      <ConfirmDialog
        isOpen={pendingRemoveId !== ""}
        onOpenChange={(open) => {
          if (!open) setPendingRemoveId("")
        }}
        heading="Remove this router?"
        body={
          pendingRemove
            ? `${pendingRemove.model} and its stored key are deleted. With no router left, the decision model agent_recommender_model names answers.`
            : ""
        }
        confirmLabel="Remove router"
        confirmVariant="danger"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          remove.mutate(pendingRemoveId, {
            onSuccess: () => setPendingRemoveId(""),
          })
        }}
      />
    </div>
  )
}
