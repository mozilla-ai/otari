import { useState } from "react"

import type {
  CreateHostedProviderRegionRequest,
  HostedProvider,
  HostedProviderRegion,
  UpdateHostedProviderRegionRequest,
} from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Disclosure } from "@/design-system/navigation/Disclosure"
import {
  ClientArgsField,
  formatClientArgs,
  parseClientArgs,
} from "@/features/providers/providerFields"
import {
  useAddHostedProviderRegion,
  useUpdateHostedProviderRegion,
} from "@/shared/api/hostedProviders"

import { DataLocationField } from "./DataLocationField"
import {
  dataLocationLabel,
  primaryLabelFor,
  primarySlugFor,
  REGION_SLUG_PATTERN,
} from "./dataLocations"
import { SupportedCountriesField } from "./SupportedCountriesField"
import { formatCountries, parseCountries } from "./supportedCountries"

// Adds a region to a hosted provider, or edits one. A region states only what
// differs from the provider: every field left blank is the provider's, which
// is why nothing but its name and id is required. The key is a choice rather
// than a blank box, because "no key of its own" and "a key not typed yet" read
// the same in an empty field.
//
// A provider with no regions serves from its own connection, and adding its
// first region keeps that connection as a region ahead of the new one, so the
// dialog asks what to call it.

type KeyMode = "provider" | "own"

const KEY_OPTIONS = [
  { value: "provider", label: "Use the provider's key" },
  { value: "own", label: "Its own key" },
] as const

interface RegionDraft {
  label: string
  slug: string
  apiBase: string
  dataLocation: string
  keyMode: KeyMode
  apiKey: string
  clientArgs: string
  supportedCountries: string
  primaryLabel: string
  primarySlug: string
}

function draftFrom(
  provider: HostedProvider,
  region: HostedProviderRegion | undefined,
): RegionDraft {
  return {
    label: region?.label ?? "",
    slug: region?.slug ?? "",
    apiBase: region?.api_base ?? "",
    dataLocation: region?.data_location ?? "",
    keyMode: region?.has_api_key ? "own" : "provider",
    apiKey: "",
    clientArgs: formatClientArgs(region?.client_args),
    supportedCountries: formatCountries(region?.supported_countries),
    primaryLabel: primaryLabelFor(provider),
    primarySlug: primarySlugFor(provider),
  }
}

const SLUG_ERROR =
  "Lowercase letters, digits and hyphens, starting with a letter or digit, at most 32."

export function RegionFormDialog({
  provider,
  region,
  isOpen,
  onClose,
}: {
  provider: HostedProvider
  /** The region being edited, or undefined when adding one. */
  region?: HostedProviderRegion
  isOpen: boolean
  onClose: () => void
}) {
  const add = useAddHostedProviderRegion(provider.name)
  const update = useUpdateHostedProviderRegion(provider.name)
  const [draft, setDraft] = useState<RegionDraft>(() =>
    draftFrom(provider, region),
  )
  // The region as the dialog opened on it. The save is built against this,
  // so a refresh landing while the dialog is open is not undone by a save
  // that never touched the field it changed.
  const [opened] = useState(region)
  const { isDirty } = useDirtySnapshot(draft)

  const isEditing = region !== undefined
  const existing = provider.regions ?? []
  const isFirst = !isEditing && existing.length === 0
  const label = draft.label.trim()
  const slug = draft.slug.trim()
  const primarySlug = draft.primarySlug.trim()
  const takenSlugs = new Set([
    ...existing.map((row) => row.slug),
    ...(isFirst ? [primarySlug] : []),
  ])
  const slugError = isEditing
    ? undefined
    : slug !== "" && !REGION_SLUG_PATTERN.test(slug)
      ? SLUG_ERROR
      : takenSlugs.has(slug)
        ? "Another region already has this id."
        : undefined
  const primarySlugError =
    isFirst && primarySlug !== "" && !REGION_SLUG_PATTERN.test(primarySlug)
      ? SLUG_ERROR
      : undefined
  const parsedClientArgs = parseClientArgs(draft.clientArgs)
  const countries = parseCountries(draft.supportedCountries)
  // A key of its own that is not stored yet has to be typed.
  const needsKey =
    draft.keyMode === "own" && !region?.has_api_key && draft.apiKey === ""
  const canSubmit =
    label !== "" &&
    (isEditing || (slug !== "" && slugError === undefined)) &&
    (!isFirst ||
      (draft.primaryLabel.trim() !== "" &&
        primarySlug !== "" &&
        primarySlugError === undefined)) &&
    !needsKey &&
    parsedClientArgs.ok &&
    countries.ok

  const providerLocation = provider.data_location
    ? dataLocationLabel(provider.data_location)
    : ""

  const createBody = (): CreateHostedProviderRegionRequest | null => {
    if (!parsedClientArgs.ok || !countries.ok) return null
    const apiBase = draft.apiBase.trim()
    return {
      slug,
      label,
      ...(draft.keyMode === "own" ? { api_key: draft.apiKey } : {}),
      ...(apiBase ? { api_base: apiBase } : {}),
      ...(draft.dataLocation ? { data_location: draft.dataLocation } : {}),
      ...(parsedClientArgs.value
        ? { client_args: parsedClientArgs.value }
        : {}),
      ...(countries.value === null
        ? {}
        : { supported_countries: countries.value }),
      ...(isFirst
        ? {
            primary_slug: primarySlug,
            primary_label: draft.primaryLabel.trim(),
          }
        : {}),
    }
  }

  // Only what changed: a field left alone keeps what is stored, and null puts
  // one back on the provider's value.
  const updateBody = (
    current: HostedProviderRegion,
  ): UpdateHostedProviderRegionRequest | null => {
    if (!parsedClientArgs.ok || !countries.ok) return null
    const apiBase = draft.apiBase.trim()
    const key =
      draft.keyMode === "provider"
        ? current.has_api_key
          ? { api_key: null }
          : {}
        : draft.apiKey === ""
          ? {}
          : { api_key: draft.apiKey }
    return {
      ...(label === current.label ? {} : { label }),
      ...key,
      ...(apiBase === (current.api_base ?? "")
        ? {}
        : { api_base: apiBase || null }),
      ...(draft.dataLocation === (current.data_location ?? "")
        ? {}
        : { data_location: draft.dataLocation || null }),
      // Echoed masked entries keep their stored values, but an untouched box
      // is not sent at all.
      ...(draft.clientArgs.trim() ===
      formatClientArgs(current.client_args).trim()
        ? {}
        : { client_args: parsedClientArgs.value }),
      ...(draft.supportedCountries.trim() ===
      formatCountries(current.supported_countries)
        ? {}
        : { supported_countries: countries.value }),
    }
  }

  const submit = () => {
    if (!canSubmit) return
    if (region && opened) {
      const body = updateBody(opened)
      if (body)
        update.mutate({ slug: region.slug, body }, { onSuccess: onClose })
      return
    }
    const body = createBody()
    if (body) add.mutate(body, { onSuccess: onClose })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={isEditing ? `Edit ${region.label}` : "Add region"}
      description={
        isEditing
          ? `Requests pin this region as ${region.selector}:<model>. A field left blank uses ${provider.name}'s value.`
          : `Another place ${provider.name} serves from, with its own endpoint, key or data location. A field left blank uses the provider's value.`
      }
      submitLabel={isEditing ? "Save" : "Add region"}
      onSubmit={submit}
      isPending={add.isPending || update.isPending}
      isSubmitDisabled={!canSubmit}
      isDirty={isDirty}
      error={add.error ?? update.error}
    >
      {isFirst ? (
        <fieldset className="flex flex-col gap-3">
          <legend className="text-body">Current connection</legend>
          <p className="text-caption">
            {provider.name}'s current key and endpoint become its first region,
            tried before the one added here.
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field
              label="Current connection's name"
              value={draft.primaryLabel}
              onChange={(primaryLabel) => setDraft({ ...draft, primaryLabel })}
              isRequired
            />
            <Field
              label="Current connection's id"
              value={draft.primarySlug}
              onChange={(value) => setDraft({ ...draft, primarySlug: value })}
              isRequired
              description={`Pinned as ${provider.name}@${primarySlug || "<id>"}:<model>.`}
              isInvalid={primarySlugError !== undefined}
              errorMessage={primarySlugError}
            />
          </div>
        </fieldset>
      ) : null}
      <Field
        label="Name"
        value={draft.label}
        onChange={(value) => setDraft({ ...draft, label: value })}
        placeholder="Southeast Asia"
        isRequired
      />
      {isEditing ? null : (
        <Field
          label="Region id"
          value={draft.slug}
          onChange={(value) => setDraft({ ...draft, slug: value })}
          placeholder="sg"
          isRequired
          description={`Requests pin it as ${provider.name}@${slug || "<id>"}:<model>. It cannot be changed later.`}
          isInvalid={slugError !== undefined}
          errorMessage={slugError}
        />
      )}
      <Field
        label="Endpoint URL"
        value={draft.apiBase}
        onChange={(apiBase) => setDraft({ ...draft, apiBase })}
        placeholder={provider.api_base ?? "The provider's endpoint"}
        description="Leave blank to use the provider's endpoint."
      />
      <DataLocationField
        value={draft.dataLocation}
        onChange={(dataLocation) => setDraft({ ...draft, dataLocation })}
        placeholder={providerLocation || "Not declared"}
        description={
          providerLocation
            ? `Leave blank to use the provider's: ${providerLocation}.`
            : "Where this region keeps request data. Leave blank to use the provider's, which is not declared."
        }
      />
      <RadioGroup
        label="API key"
        orientation="horizontal"
        value={draft.keyMode}
        onChange={(value) =>
          setDraft({
            ...draft,
            keyMode: value === "own" ? "own" : "provider",
            apiKey: "",
          })
        }
        options={KEY_OPTIONS}
      />
      {draft.keyMode === "own" ? (
        <SecretField
          label="Region's API key"
          value={draft.apiKey}
          onChange={(apiKey) => setDraft({ ...draft, apiKey })}
          isRequired={!region?.has_api_key}
          placeholder={
            region?.has_api_key
              ? "Leave blank to keep the stored key"
              : undefined
          }
          description={
            region?.has_api_key && region.api_key_last4
              ? `The stored key ends in ${region.api_key_last4}. Encrypted at rest and never shown again.`
              : "Encrypted at rest and never shown again; only the last 4 characters come back."
          }
        />
      ) : null}
      <Disclosure heading="More">
        <div className="flex flex-col gap-4">
          <ClientArgsField
            isEncryptedAtRest
            value={draft.clientArgs}
            onChange={(clientArgs) => setDraft({ ...draft, clientArgs })}
            error={parsedClientArgs.ok ? null : parsedClientArgs.error}
          />
          <SupportedCountriesField
            value={draft.supportedCountries}
            parsed={countries}
            placeholder="The provider's countries"
            onChange={(supportedCountries) =>
              setDraft((current) => ({ ...current, supportedCountries }))
            }
          />
        </div>
      </Disclosure>
    </FormDialog>
  )
}
