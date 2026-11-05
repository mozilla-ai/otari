import { useState } from "react"

import type {
  SearchProviderInfo,
  WorkspaceWebSearchKey,
  WorkspaceWebSearchKeys,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { ProviderOptionFields } from "@/features/webSearch/ProviderOptionFields"
import {
  buildOptions,
  sameOptions,
  splitOptions,
  WEB_TOOL_KINDS,
  type WebToolKind,
} from "@/features/webSearch/webTools"
import { useSearchProviders } from "@/shared/api/tools"
import {
  usePinWorkspaceWebSearchKey,
  useResetWorkspaceWebSearchKeys,
  useWorkspaceWebSearchKeys,
} from "@/shared/api/webSearchKeys"
import { formatNumber } from "@/shared/helpers/format"
import { useAutosave } from "@/shared/hooks/useAutosave"

// Which of its organization's search and fetch keys a workspace uses. The
// gateway takes the key the workspace pinned, else its organization's default,
// else the deployment's own tool, so the list offers the deployment's tool and
// each key. Choosing a key pins it; choosing the organization's default or the
// deployment's tool clears the pin, so the workspace follows its organization.

/** The list's value for the deployment's own tool. */
const DEPLOYMENT = "deployment"

const LABEL: Readonly<Record<WebToolKind, string>> = {
  search: "Search with",
  fetch: "Fetch with",
}

/** The deployment's row, with its tool's name when it has one. */
function deploymentLabel(tool: string | null | undefined): string {
  return tool
    ? `This deployment's default tool · ${tool}`
    : "This deployment's default tool"
}

/** What a key's row says about it beyond its name. */
function keyMarks(key: WorkspaceWebSearchKey): string {
  return [
    key.last4 ? `••••${key.last4}` : "",
    key.is_default ? "Pinned" : "",
    key.is_org_default ? "Your organization's default" : "",
    key.disabled ? "Turned off here" : "",
    key.usable ? "" : "Unreadable",
  ]
    .filter(Boolean)
    .join(" · ")
}

function KindChoice({
  kind,
  workspaceId,
  listing,
  canChoose,
}: {
  kind: WebToolKind
  workspaceId: string
  listing: WorkspaceWebSearchKeys
  canChoose: boolean
}) {
  const pin = usePinWorkspaceWebSearchKey()
  const reset = useResetWorkspaceWebSearchKeys()
  const keys = listing.data.filter((key) => key.kind === kind)
  const effective = keys.find((key) => key.is_effective)
  const orgDefault = keys.find((key) => key.is_org_default)
  // The provider the workspace's calls run on: its key's when one serves it,
  // otherwise the deployment's own tool's, empty when nothing serves it.
  const provider =
    (kind === "search" ? listing.search_provider : listing.fetch_provider) ?? ""
  const deploymentTool =
    kind === "search"
      ? listing.deployment_search_tool
      : listing.deployment_fetch_tool
  const nothing =
    kind === "search"
      ? "This deployment has no search tool, so nothing searches for this workspace."
      : "This deployment has no fetch tool, so nothing fetches for this workspace."
  // What keeps the workspace off its organization's default: a pin, or the
  // default turned off for it. A turned-off mark on another key changes nothing
  // while the workspace follows its organization, so it is kept.
  const departures = keys.filter(
    (key) => key.is_default || (key.is_org_default && key.disabled),
  )
  const followOrganization = () =>
    reset.mutate({
      workspaceId,
      keyIds: departures.map((key) => key.org_web_search_key_id),
    })

  if (keys.length === 0) {
    return (
      <SettingRow
        label={LABEL[kind]}
        help={
          provider === ""
            ? nothing
            : `${deploymentLabel(deploymentTool)}. Your organization has no ${kind} keys.`
        }
        control={null}
      />
    )
  }

  const options: RadioOption[] = [
    {
      value: DEPLOYMENT,
      label: deploymentLabel(deploymentTool),
      description: deploymentTool
        ? "Used when your organization has no default."
        : `Used when your organization has no default. This deployment has none, so nothing ${kind === "search" ? "searches" : "fetches"} with this chosen.`,
      // A workspace cannot pin the deployment's tool over an organization
      // default that serves; it can only stop pinning, which returns it to that
      // default. A default this deployment cannot decrypt is passed over, so
      // stopping the pin then lands on the deployment's tool.
      isDisabled: Boolean(orgDefault?.usable) && Boolean(effective),
    },
    ...keys.map((key) => ({
      value: key.org_web_search_key_id,
      label: `${key.provider} · ${key.name}`,
      description: keyMarks(key),
    })),
  ]

  return (
    <div className="flex flex-col gap-2 py-3 pr-4 pl-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-emphasis">{LABEL[kind]}</span>
        {canChoose && orgDefault?.usable && departures.length > 0 ? (
          <Button
            size="sm"
            variant="ghost"
            isDisabled={reset.isPending || pin.isPending}
            onPress={followOrganization}
          >
            Use the organization&rsquo;s default
          </Button>
        ) : null}
      </div>
      <RadioGroup
        label={LABEL[kind]}
        hideLabel
        value={effective?.org_web_search_key_id ?? DEPLOYMENT}
        onChange={(next) => {
          if (
            next === DEPLOYMENT ||
            next === orgDefault?.org_web_search_key_id
          ) {
            followOrganization()
          } else pin.mutate({ workspaceId, keyIds: [next] })
        }}
        options={options}
        isDisabled={!canChoose || pin.isPending || reset.isPending}
      />
      <ErrorBanner error={pin.error ?? reset.error} />
    </div>
  )
}

/**
 * The workspace's search and fetch choice, one list per kind. Read-only for a
 * member; a workspace's owners and admins, and its organization's, choose.
 */
export function WorkspaceWebSearchKeyChoice({
  workspaceId,
  canChoose,
}: {
  workspaceId: string
  canChoose: boolean
}) {
  const listing = useWorkspaceWebSearchKeys(workspaceId)
  if (!listing.data) {
    return listing.error ? (
      <div className="px-4 py-3">
        <ErrorBanner error={listing.error} />
      </div>
    ) : null
  }
  const { count, data } = listing.data
  return (
    <>
      {count > data.length ? (
        <p className="px-4 pt-3 text-caption text-subtle">
          {`These lists show the first ${formatNumber(data.length)} of your organization's ${formatNumber(count)} keys.`}
        </p>
      ) : null}
      {WEB_TOOL_KINDS.map((kind) => (
        <KindChoice
          key={kind}
          kind={kind}
          workspaceId={workspaceId}
          listing={listing.data}
          canChoose={canChoose}
        />
      ))}
    </>
  )
}

/** The provider's options a workspace may set: every one an operator need not. */
function workspaceOptions(
  provider: SearchProviderInfo | undefined,
): SearchProviderInfo | undefined {
  return provider
    ? {
        ...provider,
        options: (provider.options ?? []).filter(
          (option) => !option.operator_only,
        ),
      }
    : undefined
}

function OptionsForm({
  provider,
  stored,
  disabled,
  save,
}: {
  provider: SearchProviderInfo
  stored: Record<string, unknown>
  disabled: boolean
  save: (options: Record<string, unknown> | null) => Promise<unknown>
}) {
  const [draft, setDraft] = useState(() => splitOptions(provider, stored))
  const autosave = useAutosave()
  const built = buildOptions(provider, draft.typed, draft.extraJson)
  const isChanged = built.ok && !sameOptions(built.options, stored)
  return (
    <div className="flex flex-col gap-3 py-3 pr-4 pl-4">
      <div className="flex flex-col gap-0.5">
        <span className="text-emphasis">{`${provider.id} options`}</span>
        <p className="text-caption text-subtle">
          {`Sent with this workspace's searches on ${provider.id}. When its searches move to another provider, the options that provider does not take are ignored.`}
        </p>
      </div>
      <ProviderOptionFields
        provider={provider}
        values={draft.typed}
        errors={built.ok ? {} : built.fieldErrors}
        extraJson={draft.extraJson}
        jsonError={built.ok ? "" : built.jsonError}
        onChange={(typed) => setDraft({ ...draft, typed })}
        onExtraJsonChange={(extraJson) => setDraft({ ...draft, extraJson })}
      />
      {autosave.error ? (
        <p className="text-caption text-danger">{autosave.error}</p>
      ) : null}
      <div>
        <Button
          size="sm"
          isDisabled={disabled || !isChanged || autosave.isSaving}
          isPending={autosave.isSaving}
          onPress={() => {
            if (!built.ok) return
            const options = built.options
            void autosave.run(() =>
              save(Object.keys(options).length > 0 ? options : null),
            )
          }}
        >
          Save options
        </Button>
      </div>
    </div>
  )
}

/**
 * The workspace's own options for the provider its searches run on, typed from
 * the catalog's schema, with JSON for the rest. Nothing when nothing searches.
 */
export function WorkspaceProviderOptions({
  workspaceId,
  stored,
  disabled,
  save,
}: {
  workspaceId: string
  stored: Record<string, unknown> | null | undefined
  disabled: boolean
  save: (options: Record<string, unknown> | null) => Promise<unknown>
}) {
  const listing = useWorkspaceWebSearchKeys(workspaceId)
  const catalog = useSearchProviders("search")
  const providerId = listing.data?.search_provider ?? ""
  const provider = workspaceOptions(
    catalog.data?.find((entry) => entry.id === providerId),
  )
  if (!provider) return null
  return (
    <OptionsForm
      // Seeded once, so a new provider or a saved change starts a fresh draft.
      key={`${provider.id}-${JSON.stringify(stored ?? {})}`}
      provider={provider}
      stored={stored ?? {}}
      disabled={disabled}
      save={save}
    />
  )
}
