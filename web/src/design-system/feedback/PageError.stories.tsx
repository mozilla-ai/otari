import type { Meta, StoryObj } from "@storybook/react-vite"

import { PageError } from "./PageError"

/**
 * A failure that took the whole page rather than a band inside one.
 *
 * `PageLoading`'s counterpart, and it exists for the same reason: three places
 * reach this state (a gateway that never answered, a throw above the router,
 * and a throw inside it) and none of them should be drawing its own centered
 * box.
 *
 * `children` is the sentence about what to do next. The gateway case puts that
 * in the banner itself and the two throwing cases cannot, which is why it is
 * optional rather than required.
 */
const meta = {
  title: "Design system/Feedback/PageError",
  component: PageError,
  args: { error: new Error("Could not reach the gateway.") },
  parameters: { layout: "fullscreen" },
} satisfies Meta<typeof PageError>

export default meta

type Story = StoryObj<typeof meta>

/** The banner alone, which is what a failed bootstrap shows. */
export const Default: Story = {}

/** With the sentence a caught throw adds, since the thrown value carries none. */
export const WithGuidance: Story = {
  args: {
    error: new Error("Cannot read properties of undefined (reading 'map')"),
    children:
      "The dashboard could not finish rendering this page. Reload to try again. If it keeps happening, this gateway and the dashboard it serves may be out of step.",
  },
}

/**
 * A falsy throw. `throw ""`, `throw 0` and `throw null` are all legal and none
 * carries a message, and `ErrorBanner` renders nothing at all for a falsy
 * value, so the panel would be an empty box. The substitution happens here
 * rather than at each caller, so neither catch boundary can reach the panel
 * with an empty banner in it.
 */
export const FalsyThrow: Story = {
  render: () => (
    <div className="flex flex-col">
      <PageError error={null} />
      <PageError error="" />
      <PageError error={0} />
    </div>
  ),
}

export const Dark: Story = { globals: { theme: "dark" } }
