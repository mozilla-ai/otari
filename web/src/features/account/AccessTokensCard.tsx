import { useRef, useState } from "react"
import { FiPlus, FiTrash2 } from "react-icons/fi"

import type {
  AccessToken,
  CreateAccessTokenRequest,
  CreatedAccessToken,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { CONCEALED_SECRET, CopyField } from "@/design-system/actions/CopyField"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { Section } from "@/design-system/layout/Section"
import {
  useAccessTokens,
  useCreateAccessToken,
  useDeleteAccessToken,
} from "@/shared/api/auth"
import { formatDate, formatDateTime } from "@/shared/helpers/format"
import { resolveSnippetBaseUrl } from "@/shared/helpers/requestSnippets"

type Expiration = CreateAccessTokenRequest["expiration"]

const EXPIRATIONS: readonly { value: Expiration; label: string }[] = [
  { value: "1-hour", label: "1 hour" },
  { value: "1-day", label: "1 day" },
  { value: "7-days", label: "7 days" },
  { value: "30-days", label: "30 days" },
  { value: "90-days", label: "90 days" },
  { value: "180-days", label: "180 days" },
  { value: "1-year", label: "1 year" },
]

function TokenRow({
  token,
  onDelete,
  isBusy,
}: {
  token: AccessToken
  onDelete: () => void
  isBusy: boolean
}) {
  return (
    <li className="flex flex-col gap-3 border-b border-border py-4 last:border-b-0 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <p className="truncate text-body">
          {token.name}{" "}
          <span className="font-mono text-caption">…{token.token_suffix}</span>
        </p>
        <p className="text-caption">
          Created {formatDate(token.created_at)}
          {" · "}
          {token.last_used_at
            ? `Last used ${formatDateTime(token.last_used_at)}`
            : "Never used"}
          {" · "}
          {token.expires_at
            ? `Expires ${formatDate(token.expires_at)}`
            : "Never expires"}
        </p>
      </div>
      <RowActionRow>
        <RowAction
          icon={FiTrash2}
          label="Delete…"
          isDestructive
          isDisabled={isBusy}
          onPress={onDelete}
        />
      </RowActionRow>
    </li>
  )
}

/**
 * The token, shown once. Closing is the acknowledgement and the only way out,
 * so a stray Escape cannot lose it.
 */
function TokenCreated({
  created,
  onClose,
}: {
  created: CreatedAccessToken
  onClose: () => void
}) {
  const mcpUrl = `${resolveSnippetBaseUrl()}/mcp`
  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      isDismissable={false}
      title="Access token created"
      size="lg"
      submitLabel="I’ve saved this token"
      onSubmit={onClose}
      isPending={false}
      hideCancel
    >
      <div
        role="alert"
        aria-label="Access token created"
        className="flex flex-col gap-4"
      >
        <p className="text-sm text-muted">
          Copy this token now. It is shown only once.
        </p>
        <CopyField
          label="Access token"
          value={created.token}
          concealed={CONCEALED_SECRET}
        />
        <CopyField label="MCP server" value={mcpUrl} />
        <p className="text-sm text-muted">
          It reads what you can see on this dashboard and never changes
          anything. Send it as{" "}
          <code className="font-mono text-xs">Authorization: Bearer</code> to
          the MCP server or the management API.
        </p>
      </div>
    </FormDialog>
  )
}

/**
 * The identity's own access tokens: read-only credentials an agent or a script
 * reads the dashboard's data with, as this person.
 */
export function AccessTokensCard() {
  const tokens = useAccessTokens()
  const create = useCreateAccessToken()
  const remove = useDeleteAccessToken()
  const createButton = useRef<HTMLButtonElement>(null)

  const [isCreating, setIsCreating] = useState(false)
  const [name, setName] = useState("")
  const [expiration, setExpiration] = useState<Expiration>("90-days")
  const [created, setCreated] = useState<CreatedAccessToken>()
  const [deleting, setDeleting] = useState<AccessToken>()

  const rows = tokens.data?.data ?? []

  const submit = () => {
    const trimmed = name.trim()
    if (!trimmed || create.isPending) {
      return
    }
    create.mutate(
      { name: trimmed, expiration },
      {
        onSuccess: (result) => {
          setIsCreating(false)
          setName("")
          setExpiration("90-days")
          setCreated(result)
        },
      },
    )
  }

  const confirmDelete = () => {
    if (!deleting || remove.isPending) {
      return
    }
    remove.mutate(deleting.id, { onSuccess: () => setDeleting(undefined) })
  }

  return (
    <>
      <Section
        className="border-t border-border pt-6 pb-5"
        contentClassName="flex flex-col gap-4"
      >
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-title">Access tokens</h2>
          <Button
            ref={createButton}
            variant="ghost"
            className="shrink-0"
            onPress={() => {
              create.reset()
              setIsCreating(true)
            }}
          >
            <FiPlus aria-hidden className="size-4" />
            Create token
          </Button>
        </div>

        <p className="max-w-3xl text-sm text-muted">
          Let an agent or a script read what you can see in this organization,
          through the MCP server or the API, even after you switch to another. A
          token never changes anything.
        </p>

        <ErrorBanner error={tokens.error} />
        <ErrorBanner error={remove.error} />

        {tokens.isPending && !tokens.data ? (
          <p className="text-sm text-muted" role="status">
            Loading access tokens…
          </p>
        ) : rows.length > 0 ? (
          <ul className="flex flex-col">
            {rows.map((token) => (
              <TokenRow
                key={token.id}
                token={token}
                isBusy={remove.isPending}
                onDelete={() => {
                  setDeleting(token)
                  remove.reset()
                }}
              />
            ))}
          </ul>
        ) : tokens.isSuccess ? (
          <p className="text-sm text-muted">You have no access tokens.</p>
        ) : null}
      </Section>

      <FormDialog
        isOpen={isCreating}
        onOpenChange={setIsCreating}
        title="New access token"
        submitLabel="Create token"
        onSubmit={submit}
        isPending={create.isPending}
        error={create.error}
        isDirty={name.trim() !== ""}
        isSubmitDisabled={!name.trim()}
        returnFocusRef={createButton}
      >
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="Claude Code"
          isRequired
          autoFocus
        />
        <Select
          label="Expires after"
          value={expiration}
          onChange={(value) => setExpiration(value as Expiration)}
          options={EXPIRATIONS}
        />
      </FormDialog>

      {created ? (
        <TokenCreated created={created} onClose={() => setCreated(undefined)} />
      ) : null}

      <ConfirmDialog
        isOpen={deleting !== undefined}
        onOpenChange={(open) => {
          if (!open) {
            setDeleting(undefined)
          }
        }}
        heading="Delete this access token?"
        confirmLabel="Delete"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={confirmDelete}
        body={
          <>
            Anything using{" "}
            <strong className="text-foreground">{deleting?.name}</strong> stops
            working at once.
          </>
        }
      />
    </>
  )
}
