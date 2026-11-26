import type { Meta, StoryObj } from "@storybook/react-vite"
import { BootFailureView } from "@/app/BootFailureView"
import { BootConnecting } from "@/app/BootFrame"

/**
 * The screen the dashboard shows when the gateway has not answered its first
 * question, in the sign-in shell's frame, one story per state.
 *
 * The four retrying states share their words and sweep the card's border in
 * danger ink; the Last try row is where they differ. The states that wait for
 * the reader (a sign-in, a refusal, an unexpected answer, no gateway chosen)
 * hold the border still.
 */
const meta = {
  title: "Dashboard/Boot/BootFailureView",
  component: BootFailureView,
  parameters: { layout: "fullscreen" },
  args: {
    failedAt: new Date(2026, 10, 25, 14, 2, 31).getTime(),
    host: "otari.example.up.railway.app",
    isPending: false,
    onAction: () => {},
  },
} satisfies Meta<typeof BootFailureView>

export default meta

type Story = StoryObj<typeof meta>

export const Unreachable: Story = {
  args: {
    failure: { kind: "unreachable", reason: "no response", willRetry: true },
  },
}

export const TimedOut: Story = {
  args: {
    failure: { kind: "timedOut", reason: "no answer in 8 s", willRetry: true },
  },
}

export const ServerError: Story = {
  args: {
    failure: { kind: "serverError", reason: "HTTP 502", willRetry: true },
  },
}

export const Offline: Story = {
  args: {
    failure: {
      kind: "offline",
      reason: "offline",
      willRetry: true,
    },
  },
}

/** A pressed Retry waiting on its attempt: full ink, a spinner, no refusal. */
export const RetryPressed: Story = {
  args: {
    failure: { kind: "unreachable", reason: "no response", willRetry: true },
    isPending: true,
  },
}

/** The pressed attempt failed too. */
export const RetryPressedFailed: Story = {
  args: {
    failure: { kind: "unreachable", reason: "no response", willRetry: true },
    failedAt: new Date(2026, 10, 25, 14, 3, 2).getTime(),
    isStillNoAnswer: true,
  },
}

export const SignInRequired: Story = {
  args: {
    failure: { kind: "signInRequired", reason: "HTTP 401", willRetry: false },
  },
}

/** A redirect to a sign-in page, which the bootstrap read cannot follow to an answer. */
export const SignInRedirect: Story = {
  args: {
    failure: {
      kind: "signInRequired",
      reason: "sent to sign-in",
      willRetry: false,
    },
  },
}

export const AccessRefused: Story = {
  args: {
    failure: { kind: "accessRefused", reason: "HTTP 403", willRetry: false },
  },
}

export const UnexpectedAnswer: Story = {
  args: {
    failure: {
      kind: "unexpectedAnswer",
      reason: "HTTP 404",
      willRetry: false,
    },
  },
}

/** No gateway was asked, so there is nothing to diagnose. */
export const NoGatewayChosen: Story = {
  args: {
    failure: { kind: "unchosen", reason: "", willRetry: false },
  },
}

/**
 * What a caller that mounts before the first attempt settles sees. `index.html`
 * paints its own copy of this frame while the bundle loads.
 */
export const Loading: Story = {
  args: {
    failure: { kind: "unreachable", reason: "no response", willRetry: true },
  },
  render: () => <BootConnecting />,
}
