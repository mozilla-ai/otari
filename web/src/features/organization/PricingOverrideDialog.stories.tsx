import { Button } from "@heroui/react"
import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import type { OrganizationPricingOverride } from "@/client"
import { API_ROOT } from "@/shared/api/client"

import { PricingOverrideDialog } from "./PricingOverrideDialog"

/**
 * An organization's own rates for a model, valid over a date window.
 *
 * The windows are the interesting part: the dialog validates a new one against
 * `existing`, so two overlapping overrides for the same model cannot both be
 * saved. That check runs client-side against the list passed in here, which is why
 * `existing` is a prop rather than something the dialog fetches.
 */
function override(
  overrides: Partial<OrganizationPricingOverride> = {},
): OrganizationPricingOverride {
  return {
    id: "018f0000-0000-4000-8000-000000000001",
    organization_id: "018f0000-0000-4000-8000-0000000000ff",
    model_key: "openai:gpt-4o-mini",
    input_price_per_million: 0.15,
    output_price_per_million: 0.6,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    cache_write_1h_price_per_million: null,
    pricing_tiers: [],
    unit: "tokens",
    effective_from: "2026-01-01T00:00:00Z",
    effective_to: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  }
}

const EXISTING: OrganizationPricingOverride[] = [
  override(),
  override({
    id: "018f0000-0000-4000-8000-000000000002",
    model_key: "anthropic:claude-haiku-4-5",
    input_price_per_million: 0.8,
    output_price_per_million: 4,
    effective_from: "2026-03-01T00:00:00Z",
    effective_to: "2026-09-01T00:00:00Z",
  }),
]

// What the model-key picker offers. The catalog rather than discovery, because
// this dialog answers to an organization admin who is refused the
// deployment-operator read; `fast` is an alias, and is left out of the list.
const CATALOG = {
  object: "list",
  data: [
    "openai:gpt-4o-mini",
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-sonnet-5",
    "fast",
  ].map((id) => ({
    id,
    object: "model",
    created: 0,
    owned_by: id.split(":")[0],
    pricing_source: "none",
    deployment_managed: false,
  })),
}

// The same catalog with one model the deployment supplies the credential for,
// which is what the refusal below reads.
const CATALOG_WITH_A_MANAGED_MODEL = {
  ...CATALOG,
  data: [
    ...CATALOG.data,
    {
      id: "nebius_prod:llama-3",
      object: "model",
      created: 0,
      owned_by: "nebius_prod",
      pricing_source: "configured",
      deployment_managed: true,
    },
  ],
}

// A tenant rather than the deployment's operator, which is the audience the
// refusal is for: an operator pays the upstream bill and keeps setting the rate.
const TENANT_CONTEXT = {
  organization_member_id: "018f0000-0000-4000-8000-00000000000a",
  caller: {
    user_id: "018f0000-0000-4000-8000-00000000000b",
    email: "admin@acme.test",
    full_name: "Acme admin",
  },
  role: "admin",
  status: "active",
  organization: {
    id: "018f0000-0000-4000-8000-0000000000ff",
    name: "Acme",
    slug: "acme",
  },
  workspace_memberships: [],
  deployment_operator: false,
}

const meta = {
  title: "Dashboard/Organization/PricingOverrideDialog",
  component: PricingOverrideDialog,
  args: {
    isOpen: true,
    onOpenChange: () => {},
    existing: EXISTING,
    onSaved: () => {},
  },
  parameters: { api: { [`${API_ROOT}/models`]: CATALOG } },
} satisfies Meta<typeof PricingOverrideDialog>

export default meta

type Story = StoryObj<typeof meta>

/** Adding a new override. */
export const Add: Story = {}

/** Editing an existing one, which prefills every field including the window. */
export const Edit: Story = {
  args: { editing: EXISTING[1] },
}

/** An open-ended override: no `effective_to`, so it applies from its start onward. */
export const EditOpenEnded: Story = {
  args: { editing: EXISTING[0] },
}

/** The first override for this organization, so nothing can overlap yet. */
export const NoExistingOverrides: Story = {
  args: { existing: [] },
}

// No `Pending` or `WithError` story: this dialog owns the create and replace
// mutations, so neither state can be handed in as a prop. Both are asserted in
// this component's own tests, along with the refusal not carrying into the next
// open, which is what owning the mutation buys.

/**
 * Driven from a trigger, so the overlap validation can be exercised: try
 * `openai:gpt-4o-mini` from a date inside the open-ended window above.
 */
export const FromTrigger: Story = {
  render: (args) => {
    const [open, setOpen] = useState(false)
    return (
      <div className="flex flex-col items-start gap-3">
        <Button variant="primary" onPress={() => setOpen(true)}>
          Add override
        </Button>
        <PricingOverrideDialog
          {...args}
          isOpen={open}
          onOpenChange={setOpen}
          onSaved={() => setOpen(false)}
        />
      </div>
    )
  },
}

/**
 * A model this deployment supplies the provider key for, so its rate is the
 * catalog's rather than this organization's (otari-ai#2095).
 *
 * Type `nebius_prod:llama-3` into Model key: the field says whose rate it is and
 * the save stays disabled. The gateway refuses the write either way; this is the
 * half that stops it being offered.
 */
export const DeploymentSuppliedModel: Story = {
  parameters: {
    api: {
      [`${API_ROOT}/models`]: CATALOG_WITH_A_MANAGED_MODEL,
      [`${API_ROOT}/organizations/me`]: TENANT_CONTEXT,
    },
  },
}
