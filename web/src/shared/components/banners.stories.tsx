import type { Meta, StoryObj } from "@storybook/react-vite"

import { ApiError } from "@/shared/api/client"

import { ErrorBanner } from "./feedback/ErrorBanner"
import { InfoBanner } from "./feedback/InfoBanner"

/**
 * The two page-level banners, in one file because they are read as a pair when
 * deciding which one a page owes: `ErrorBanner` reports a request that failed,
 * `InfoBanner` states a standing condition.
 */
// Required props on the meta, so a story that supplies its own `render` still
// satisfies the component's contract without restating them.
const meta = {
  title: "Shared/Banners",
  component: ErrorBanner,
  args: { error: null },
} satisfies Meta<typeof ErrorBanner>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Every branch of `errorMessage`. An `ApiError` and an `Error` both carry their
 * own message through; anything else the gateway or a library threw becomes
 * "Something went wrong.", because an unrecognized throw must not put internals
 * in front of an operator.
 */
export const Errors: Story = {
  render: () => (
    <div className="flex w-[36rem] flex-col gap-3">
      <ErrorBanner
        error={
          new ApiError(403, "The master key has been retired as a sign-in.")
        }
      />
      <ErrorBanner
        error={new ApiError(0, "Network error: could not reach the gateway.")}
      />
      <ErrorBanner
        error={new Error("Pricing refresh failed for 3 of 42 models.")}
      />
      <ErrorBanner error="a bare string nobody should see" />
    </div>
  ),
}

/** A falsy error renders nothing, so a page can mount it unconditionally. */
export const NoError: Story = {
  render: () => (
    <div className="flex w-[36rem] flex-col gap-2">
      <ErrorBanner error={null} />
      <p className="text-caption">
        Nothing above this line: `ErrorBanner` returns null for a falsy error.
      </p>
    </div>
  ),
}

export const Info: Story = {
  render: () => (
    <div className="flex w-[36rem] flex-col gap-3">
      <InfoBanner>
        Model discovery is off, so this catalog lists only the models priced by
        hand.
      </InfoBanner>
      <InfoBanner tone="warning">
        3 models have no price. Requests routed to them are refused while
        <code className="font-mono"> require_pricing </code> is on.
      </InfoBanner>
    </div>
  ),
}
