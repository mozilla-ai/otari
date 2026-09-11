import { Button } from "@heroui/react"
import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import type { OrganizationPricingOverride } from "@/client"

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

const meta = {
  title: "Dashboard/Organization/PricingOverrideDialog",
  component: PricingOverrideDialog,
  args: {
    isOpen: true,
    onOpenChange: () => {},
    existing: EXISTING,
    onSaved: () => {},
  },
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

// No `Pending` or `WithError` story any more. This dialog owns the create and
// replace mutations (they live below the caller's key, so a refusal cannot
// greet the next open), so neither state can be handed in as a prop. Both are
// asserted where they are now produced, in `RateOverridesCard`'s page tests.

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
