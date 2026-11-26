import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { Definition } from "@/features/rules/Definition"
import { Fact } from "@/features/rules/Fact"
import { LimitCell } from "@/features/rules/LimitCell"
import { ModeChip } from "@/features/rules/ModeChip"
import { RefusedRequests } from "@/features/rules/RefusedRequests"
import {
  type PendingAction,
  RuleActionDialog,
} from "@/features/rules/RuleActionDialog"
import { RulePeople } from "@/features/rules/RulePeople"
import { RULE_TYPE_LABEL } from "@/features/rules/rules"
import { ScopeCell } from "@/features/rules/ScopeCell"
import { useRules } from "@/features/rules/useRules"
import { BackLink } from "@/shared/components/BackLink"
import { Band } from "@/shared/components/Band"
import { NotYet } from "@/shared/components/NotYet"
import { formatDate, formatUsd } from "@/shared/helpers/format"

/**
 * One rule: what it is and who it reaches, the people under it, the refusals
 * its subject met, and what the control plane stores for it. Owners and
 * admins change a limit from here; anyone else reads the rule as the server
 * gives it to them.
 */
export function RulePage({ ruleId }: { ruleId: string }) {
  const { rules, isManager, isLoading, error } = useRules()
  const [pending, setPending] = useState<PendingAction>()
  const rule = rules.find((one) => one.id === ruleId)

  if (!rule) {
    if (isLoading) return <PageLoading />
    if (error) {
      return (
        <>
          <BackLink to="/rules" label="Rules" />
          <ErrorBanner error={error} />
        </>
      )
    }
    return (
      <>
        <BackLink to="/rules" label="Rules" />
        <EmptyState
          title="No rule at this link"
          description="It may have been removed."
        />
      </>
    )
  }

  const spend = rule.spend
  const target = spend?.target
  const teamDefault = isManager ? rule.teamDefault : undefined
  // A single ceiling's spend has no people table to live in.
  const hasPeriod = isManager && spend !== undefined && !spend.isEach
  const action = target ? (
    <Button onPress={() => setPending({ action: "change-limit", rule })}>
      Change spend limit
    </Button>
  ) : teamDefault ? (
    <Button onPress={() => setPending({ action: "change-default", rule })}>
      Change the limit for everyone in {teamDefault.team}
    </Button>
  ) : undefined

  return (
    <>
      <BackLink to="/rules" label="Rules" />
      <div className="otari-rules-intro is-stacked">
        <PageIntro title={rule.name} action={action} />
      </div>
      <ErrorBanner error={error} />

      <Section
        className="border-y border-border"
        // Each fact as wide as it reads, spread across the row, so a long
        // Applies to is not cut to fit an equal share.
        contentClassName={`grid grid-cols-2 gap-x-8 gap-y-5 py-5 md:justify-between ${hasPeriod ? "md:grid-cols-[repeat(6,auto)]" : "md:grid-cols-[repeat(5,auto)]"}`}
      >
        <Fact label="Type">{RULE_TYPE_LABEL[rule.type]}</Fact>
        <Fact label="Applies to">
          <ScopeCell scope={rule.scope} />
        </Fact>
        {rule.type === "data" ? null : (
          <Fact label="Limit">
            <LimitCell rule={rule} />
          </Fact>
        )}
        <Fact label="Enforced in">Gateway</Fact>
        <Fact label="Mode">
          <ModeChip mode={rule.mode} />
        </Fact>
        {hasPeriod && spend.spent !== undefined ? (
          <Fact label="This period">
            <span className="flex flex-col gap-1.5">
              <span className="tabular-nums">
                {formatUsd(spend.spent)}
                <span className="text-muted"> of {formatUsd(spend.max)}</span>
              </span>
              <SpendMeter
                spent={spend.spent}
                allocated={spend.max}
                ariaLabel={`${rule.name} spent`}
              />
            </span>
          </Fact>
        ) : null}
      </Section>

      {!isManager && rule.type === "spend" ? (
        <p className="text-caption">
          Only owners and admins can see how much each person has spent against
          this limit.
        </p>
      ) : null}

      {rule.people ? <RulePeople rule={rule} isManager={isManager} /> : null}

      {rule.type === "spend" || rule.type === "access" ? (
        <RefusedRequests rule={rule} />
      ) : (
        <Band title="Decision log">
          <NotYet>
            Every decision this rule made would be listed here, each opening its
            session. The gateway does not record guardrail decisions per rule
            yet.
          </NotYet>
        </Band>
      )}

      {isManager && rule.definition !== undefined ? (
        <Definition name={rule.name} definition={rule.definition} />
      ) : null}

      {rule.createdAt ? (
        <p className="text-caption">
          Created {formatDate(rule.createdAt)}
          {rule.updatedAt
            ? `, last changed ${formatDate(rule.updatedAt)}.`
            : "."}
        </p>
      ) : null}

      {pending ? (
        <RuleActionDialog
          pending={pending}
          onClose={() => setPending(undefined)}
        />
      ) : null}
    </>
  )
}
