import type { Meta, StoryObj } from "@storybook/react-vite"

import { PricingWarning } from "./PricingWarning"

/**
 * The shell-level warning that requests are being refused for want of a price.
 *
 * It reads two things and only shows itself when both agree there is a problem:
 * `require_pricing` is on in the gateway settings, and the last hour actually
 * produced failures. That pairing is the point, `require_pricing` alone is a
 * normal, healthy configuration and is not worth a banner.
 *
 * It is also dismissible in-session, and offers to turn `require_pricing` off,
 * which is why it owns a mutation as well as its queries.
 */
function settings(overrides: Record<string, unknown> = {}) {
  return {
    version: "0.9.1",
    mode: "standalone",
    require_pricing: true,
    model_discovery: true,
    // False, because the banner needs BOTH: `require_pricing` on and default
    // pricing off. With default pricing on, a new model is metered at public
    // rates instead of being refused, so there is nothing to warn about.
    default_pricing: false,
    master_key_source: "config",
    secret_key_configured: true,
    config: {},
    ...overrides,
  }
}

const meta = {
  title: "Dashboard/Models/PricingWarning",
  component: PricingWarning,
  parameters: { layout: "padded" },
} satisfies Meta<typeof PricingWarning>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Requests are being refused. Press the dismiss control, or the offer to turn
 * `require_pricing` off.
 */
export const RequestsBeingRefused: Story = {
  parameters: {
    api: {
      "/v1/settings": settings(),
      "/v1/usage/count": { total: 142 },
    },
  },
  render: () => (
    <div className="w-[44rem]">
      <PricingWarning />
    </div>
  ),
}

/**
 * `require_pricing` is on and no request has failed yet. The banner still shows
 * (the configuration alone is the alarm) but drops the failure sentence, so it
 * reads as a config note rather than an incident.
 */
export const NoFailuresYet: Story = {
  parameters: {
    api: {
      "/v1/settings": settings(),
      "/v1/usage/count": { total: 0 },
    },
  },
  render: () => (
    <div className="w-[44rem]">
      <PricingWarning />
    </div>
  ),
}

/**
 * `require_pricing` is off, so an unpriced model is served rather than refused.
 * The banner renders nothing at all, that restraint is half the component's job,
 * so it is worth a story of its own.
 */
export const PricingNotRequired: Story = {
  parameters: {
    api: {
      "/v1/settings": settings({ require_pricing: false }),
      "/v1/usage/count": { total: 142 },
    },
  },
  render: () => (
    <div className="flex w-[44rem] flex-col gap-2">
      <PricingWarning />
      <p className="text-caption">Nothing above this line, deliberately.</p>
    </div>
  ),
}

/**
 * Default pricing is on, the other reason the banner stays quiet: a model with no
 * explicit price is metered at public rates rather than refused.
 */
export const DefaultPricingCoversIt: Story = {
  parameters: {
    api: {
      "/v1/settings": settings({ default_pricing: true }),
      "/v1/usage/count": { total: 142 },
    },
  },
  render: () => (
    <div className="flex w-[44rem] flex-col gap-2">
      <PricingWarning />
      <p className="text-caption">Nothing above this line, deliberately.</p>
    </div>
  ),
}

/** A large failure count, where the number has to stay readable. */
export const ManyFailures: Story = {
  parameters: {
    api: {
      "/v1/settings": settings(),
      "/v1/usage/count": { total: 48_912 },
    },
  },
  render: () => (
    <div className="w-[44rem]">
      <PricingWarning />
    </div>
  ),
}
