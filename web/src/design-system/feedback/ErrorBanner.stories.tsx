import type { Meta, StoryObj } from "@storybook/react-vite"

import { ErrorBanner } from "./ErrorBanner"

/**
 * The banner a page mounts for a request that failed.
 *
 * Its whole contract is `errorMessage`, which has one branch: an `Error` of any
 * kind shows its own message, and anything else shows "Something went wrong."
 * That second case is not a fallback nobody hits, it is the boundary that keeps
 * a rejected fetch body or a library's bare string off the screen.
 */
const meta = {
  title: "Design system/Feedback/ErrorBanner",
  component: ErrorBanner,
  args: { error: null },
} satisfies Meta<typeof ErrorBanner>

export default meta

type Story = StoryObj<typeof meta>

/** An `Error` carries its own message through, whatever subclass threw it. */
export const Default: Story = {
  args: { error: new Error("The master key has been retired as a sign-in.") },
}

/**
 * Both branches side by side. The bare string is the one that matters: it is
 * shown here being *withheld*, because a thrown non-`Error` is nearly always an
 * internal the operator must not read.
 */
export const EveryBranch: Story = {
  render: () => (
    <div className="flex w-[36rem] flex-col gap-3">
      <ErrorBanner error={new Error("Could not reach the gateway.")} />
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
        Nothing above this line: ErrorBanner returns null for a falsy error.
      </p>
    </div>
  ),
}
