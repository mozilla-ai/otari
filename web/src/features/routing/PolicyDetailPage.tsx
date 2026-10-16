import { Link, useNavigate } from "@tanstack/react-router"

import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { useUrlValue } from "@/shared/helpers/urlState"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { FallbackChainGroup } from "./FallbackChainGroup"
import { GuardrailsGroup } from "./GuardrailsGroup"
import { MainSelectionGroup } from "./MainSelectionGroup"
import { PolicyNameGroup } from "./PolicyNameGroup"
import { PolicyPreview } from "./PolicyPreview"
import { PolicyReadOnly } from "./PolicyReadOnly"
import { draftFromSpec, fallthroughOf, isRouted } from "./policyDraft"
import { isEditableInForm, rowKeyOf } from "./policyModel"
import { TierDownGroup } from "./TierDownGroup"
import { useRoutingRows } from "./useRoutingRows"

/**
 * One policy, section by section, where everything a policy can combine is
 * edited: its main selection, its fallback chain, a budget tier-down and
 * guardrails. Each section saves on its own.
 *
 * Takes the name as a prop and reads the scope from `?user=`, since the same
 * name can exist once for every caller and once per user.
 */
export function PolicyDetailPage({ name }: { name: string }) {
  const userId = useUrlValue("user")
  const navigate = useNavigate()
  const { deployment_type } = useDeployment()
  const {
    rows,
    isLoading,
    isFetching,
    error,
    isOperator,
    canEdit,
    writeWorkspaceId,
  } = useRoutingRows()
  const row = rows.find(
    (entry) => entry.name === name && (entry.user_id ?? "") === userId,
  )

  const back = (
    <nav aria-label="Breadcrumb" className="text-caption">
      <Link to="/routing" className="text-link hover:text-link-hover">
        ← Routing
      </Link>
    </nav>
  )

  if (row === undefined) {
    if (isLoading || isFetching) return <PageLoading label="Loading policy…" />
    return (
      <div className="flex flex-col gap-6">
        {back}
        <ErrorBanner error={error} />
        {error ? null : (
          <EmptyState
            title="Policy not found"
            description="It may have been renamed or deleted, or it belongs to another workspace."
          />
        )}
      </div>
    )
  }

  const isConfig = row.source === "config"
  const isRepresentable = isEditableInForm(row.spec)
  const isEditable = canEdit && !isConfig && isRepresentable
  const stored = draftFromSpec(row.spec)
  const sectionProps = {
    row,
    stored,
    isDeploymentWide: isOperator,
    fallbackWorkspaceId: writeWorkspaceId,
  }
  // Each section remounts when its own part of the stored policy changes, so
  // a save re-seeds it, and a save elsewhere leaves its draft alone.
  const keyOf = (section: string, part: unknown) =>
    `${rowKeyOf(row)}:${section}:${JSON.stringify(part)}`

  return (
    <div className="flex flex-col gap-6">
      {back}
      <PageIntro title={row.name}>
        {row.kind === "alias" ? "An alias for one model. " : null}
        Callers send{" "}
        <code className="font-mono text-caption">model: "{row.name}"</code>.
        Applies to{" "}
        {row.user_id ? (
          <code className="font-mono text-caption">{row.user_id}</code>
        ) : (
          "every caller"
        )}
        .
      </PageIntro>

      <ErrorBanner error={error} />

      {deployment_type === "standalone" ? null : (
        <InfoBanner>
          Policies are a standalone feature: they apply only on a gateway that
          serves requests itself. A hybrid gateway receives its attempt plan
          from the connected control plane.
        </InfoBanner>
      )}

      {isConfig ? (
        <InfoBanner>
          Set in config.yml, so it is read-only here. Change it in the file.
        </InfoBanner>
      ) : !isRepresentable ? (
        <InfoBanner>
          Uses options this page cannot show yet. Edit it through the API so
          nothing is lost.
        </InfoBanner>
      ) : !canEdit ? (
        <InfoBanner>Your organization's admins manage this policy.</InfoBanner>
      ) : null}

      {isRepresentable ? (
        <PolicyPreview
          label="Where requests go"
          name={row.name}
          target={fallthroughOf(stored)}
          backups={stored.backups}
          pool={
            isRouted(stored)
              ? {
                  backend: stored.pool.backend,
                  size: stored.pool.models.length,
                }
              : undefined
          }
          tierDowns={stored.tierDowns}
        />
      ) : null}

      {!isEditable ? (
        <PolicyReadOnly row={row} />
      ) : (
        <>
          <PolicyNameGroup
            key={rowKeyOf(row)}
            row={row}
            isDeploymentWide={isOperator}
            fallbackWorkspaceId={writeWorkspaceId}
            onRenamed={(next) =>
              void navigate({
                to: "/routing/$name",
                params: { name: next },
                search: userId ? { user: userId } : {},
                replace: true,
              })
            }
          />
          <MainSelectionGroup
            key={keyOf("main", [stored.target, stored.pool])}
            {...sectionProps}
          />
          {row.kind === "alias" ? (
            <InfoBanner>
              An alias holds one target. To add backups, a budget tier-down or
              guardrails, delete it and create a policy with the same name.
            </InfoBanner>
          ) : (
            <>
              <FallbackChainGroup
                key={keyOf("backups", stored.backups)}
                {...sectionProps}
              />
              <TierDownGroup
                key={keyOf("tier-downs", stored.tierDowns)}
                {...sectionProps}
              />
              <GuardrailsGroup
                key={keyOf("guardrails", stored.guardrails)}
                {...sectionProps}
              />
            </>
          )}
          <p className="text-caption">
            A saved change is in effect for new requests within 30s.
          </p>
        </>
      )}
    </div>
  )
}
