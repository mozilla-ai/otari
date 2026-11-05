import type { Meta, StoryObj } from "@storybook/react-vite"

import { InfoBanner } from "./InfoBanner"

/**
 * A standing condition, stated between two rules rather than in a tinted box.
 *
 * Read as a pair with `ErrorBanner`, which is what decides which one a page
 * owes: `ErrorBanner` reports a request that failed, `InfoBanner` states
 * something that is simply true about this deployment. Both tones keep the same
 * muted prose, because the dot says "worth noticing" and the words say what.
 */
const meta = {
  title: "Design system/Feedback/InfoBanner",
  component: InfoBanner,
  args: {
    children:
      "Model discovery is off, so this catalog lists only the models priced by hand.",
  },
} satisfies Meta<typeof InfoBanner>

export default meta

type Story = StoryObj<typeof meta>

/** The default: a ceiling this deployment has, not a problem with it. */
export const Default: Story = {}

/** `warning` keeps the prose and swaps the dot. Nothing else changes. */
export const Warning: Story = {
  args: {
    tone: "warning",
    children:
      "3 models have no price. Requests routed to them are refused while require_pricing is on.",
  },
}

/**
 * `onDismiss` adds a close control at the row's end. Only for a line that is not
 * a fact about the deployment, so putting it away hides nothing still true; the
 * caller decides what dismissal means and remembers it.
 */
export const Dismissible: Story = {
  args: {
    onDismiss: () => undefined,
    dismissLabel: "Dismiss the invitation to star Otari",
    children: (
      <>
        Otari is open source. A star helps other developers find it.{" "}
        <a
          href="https://github.com/mozilla-ai/otari"
          className="text-link underline underline-offset-2 hover:decoration-2"
        >
          Star Otari on GitHub
        </a>
      </>
    ),
  },
}

export const BothTones: Story = {
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
