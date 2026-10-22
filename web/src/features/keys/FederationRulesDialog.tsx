import { useState } from "react"
import { FiTerminal, FiTrash2 } from "react-icons/fi"
import type { ApiKey, FederationRule } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { CodeBlock } from "@/design-system/content/CodeBlock"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { Field } from "@/design-system/forms/Field"
import { Badge } from "@/design-system/indicators/Badge"
import {
  useAddFederationRule,
  useDeleteFederationRule,
  useFederationRules,
} from "@/shared/api/apiKeys"
import { useOrganizationContext } from "@/shared/api/organizations"
import { resolveSnippetBaseUrl } from "@/shared/helpers/requestSnippets"

const GOOGLE_ISSUER = "https://accounts.google.com"
// Where a workload's refresher writes its identity token; any path works, as
// long as the environment and the refresher agree.
const TOKEN_FILE = "/var/run/otari/identity-token"

/**
 * The environment that points Claude Code or an Anthropic SDK at one rule.
 *
 * The base URL is the gateway's `/api`, not its API root: the SDKs append
 * `/v1/...` themselves. The organization is left out when it is not known yet,
 * since a workload may omit it and a wrong one is refused.
 */
export function workloadEnvironment({
  baseUrl,
  ruleId,
  organizationId,
}: {
  baseUrl: string
  ruleId: string
  organizationId: string
}): string {
  return [
    "unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN",
    `export ANTHROPIC_BASE_URL="${baseUrl}/api"`,
    `export ANTHROPIC_FEDERATION_RULE_ID="${ruleId}"`,
    ...(organizationId
      ? [`export ANTHROPIC_ORGANIZATION_ID="${organizationId}"`]
      : []),
    `export ANTHROPIC_IDENTITY_TOKEN_FILE="${TOKEN_FILE}"`,
  ].join("\n")
}

function ruleIdentity(rule: FederationRule): string {
  return rule.email ?? rule.subject ?? rule.id
}

/**
 * The identities that may use a key without its secret, and the form that adds one.
 *
 * A rule matches only the token exchange that names its id, so the setup block
 * is per rule: the newest by default, or the one whose setup was asked for.
 */
export function FederationRulesDialog({
  apiKey,
  onClose,
}: {
  apiKey: ApiKey
  onClose: () => void
}) {
  const rules = useFederationRules(apiKey.id)
  const addRule = useAddFederationRule(apiKey.id)
  const deleteRule = useDeleteFederationRule(apiKey.id)
  const organization = useOrganizationContext()
  const baseUrl = resolveSnippetBaseUrl()

  const [issuer, setIssuer] = useState(GOOGLE_ISSUER)
  const [audience, setAudience] = useState(baseUrl)
  const [email, setEmail] = useState("")
  const [subject, setSubject] = useState("")
  const [shownRuleId, setShownRuleId] = useState("")
  const [pendingDelete, setPendingDelete] = useState<FederationRule>()

  const list = rules.data ?? []
  const shown = list.find((rule) => rule.id === shownRuleId) ?? list.at(-1)
  const isComplete =
    issuer.trim() !== "" &&
    audience.trim() !== "" &&
    (email.trim() !== "" || subject.trim() !== "")
  const isDirty =
    issuer !== GOOGLE_ISSUER ||
    audience !== baseUrl ||
    email !== "" ||
    subject !== ""

  const submit = () => {
    addRule.mutate(
      {
        issuer: issuer.trim(),
        audience: audience.trim(),
        email: email.trim() || null,
        subject: subject.trim() || null,
      },
      {
        onSuccess: (rule) => {
          setEmail("")
          setSubject("")
          setShownRuleId(rule.id)
        },
      },
    )
  }

  return (
    <>
      <FormDialog
        isOpen
        onOpenChange={(open) => {
          if (!open) onClose()
        }}
        size="lg"
        title="Workload identity"
        description={<code>{apiKey.key_name ?? apiKey.id}</code>}
        submitLabel="Add rule"
        onSubmit={submit}
        isPending={addRule.isPending}
        isSubmitDisabled={!isComplete}
        isDirty={isDirty}
        error={addRule.error ?? rules.error}
      >
        <p className="text-sm text-muted">
          A workload on Google Cloud, or on any OpenID Connect platform, can use
          this key with its own identity token instead of the secret. Each rule
          names one identity the key trusts.
        </p>
        {rules.isPending && !rules.data ? (
          <Skeleton className="h-16 w-full" />
        ) : list.length > 0 ? (
          <div className="flex flex-col gap-4">
            <ul
              aria-label="Rules"
              className="flex flex-col divide-y divide-border border-y border-border"
            >
              {list.map((rule) => (
                <li key={rule.id} className="flex items-start gap-3 py-3">
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <p className="break-all text-body">{ruleIdentity(rule)}</p>
                    <p className="break-all text-caption">
                      {rule.issuer} · {rule.audience}
                    </p>
                    {rule.member_managed ? null : (
                      <Badge tone="muted">Set by the operator</Badge>
                    )}
                  </div>
                  <div className="shrink-0">
                    <RowActionRow>
                      <RowAction
                        icon={FiTerminal}
                        label="Show setup"
                        ariaLabel={`Show setup for ${ruleIdentity(rule)}`}
                        onPress={() => setShownRuleId(rule.id)}
                      />
                      <RowAction
                        icon={FiTrash2}
                        label="Delete rule"
                        ariaLabel={`Delete rule for ${ruleIdentity(rule)}`}
                        onPress={() => setPendingDelete(rule)}
                      />
                    </RowActionRow>
                  </div>
                </li>
              ))}
            </ul>
            {shown ? (
              <div className="flex flex-col gap-2">
                <p className="text-sm text-muted">
                  Run Claude Code or an Anthropic SDK with this environment, and
                  keep a fresh identity token for{" "}
                  <span className="break-all">{ruleIdentity(shown)}</span> in
                  the token file.
                </p>
                <CodeBlock
                  label="Environment"
                  isBounded
                  value={workloadEnvironment({
                    baseUrl,
                    ruleId: shown.id,
                    organizationId: organization.data?.organization.id ?? "",
                  })}
                />
              </div>
            ) : null}
          </div>
        ) : (
          <p className="text-sm text-muted">No rules yet.</p>
        )}
        <Field
          label="Issuer"
          value={issuer}
          onChange={setIssuer}
          isRequired
          description="The token's issuer. Google Cloud's is https://accounts.google.com."
        />
        <Field
          label="Audience"
          value={audience}
          onChange={setAudience}
          isRequired
          description="The audience the workload asks its platform to put in the token."
        />
        <Field
          label="Service account email"
          value={email}
          onChange={setEmail}
          placeholder="hackbot@my-project.iam.gserviceaccount.com"
          description="The token's verified email. Give this, a subject, or both."
        />
        <Field
          label="Subject"
          value={subject}
          onChange={setSubject}
          description="The token's subject, such as a service account's numeric id."
        />
      </FormDialog>
      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDelete(undefined)
            deleteRule.reset()
          }
        }}
        heading="Delete this rule?"
        body={
          pendingDelete
            ? `${ruleIdentity(pendingDelete)} stops authenticating as this key on its next request, access tokens it already holds included.`
            : ""
        }
        confirmLabel="Delete rule"
        isPending={deleteRule.isPending}
        error={deleteRule.error}
        onConfirm={() => {
          if (!pendingDelete) return
          deleteRule.mutate(pendingDelete.id, {
            onSuccess: () => {
              setPendingDelete(undefined)
              deleteRule.reset()
            },
          })
        }}
      />
    </>
  )
}
