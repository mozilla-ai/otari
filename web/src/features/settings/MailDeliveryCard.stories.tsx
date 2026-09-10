import type { Meta, StoryObj } from "@storybook/react-vite"
import { API_ROOT } from "@/shared/api/client"
import { MailDeliveryCard } from "./MailDeliveryCard"

/**
 * Mail transport status, and a send-a-test-message form.
 *
 * This one owns its query, so its stories declare a stub gateway through
 * `parameters.api` (see `.storybook/apiMock.tsx`). The stub replaces `fetch`, so
 * the card's real `useMailSettings` hook, real query key and real error handling
 * are what run, only the network is faked, which is the same boundary the Vitest
 * suite mocks.
 */
const meta = {
  title: "Dashboard/Settings/MailDeliveryCard",
  component: MailDeliveryCard,
  parameters: { layout: "padded" },
} satisfies Meta<typeof MailDeliveryCard>

export default meta

type Story = StoryObj<typeof meta>

/** A working SMTP transport. */
export const Ready: Story = {
  parameters: {
    api: {
      [`${API_ROOT}/settings/mail`]: {
        enabled: true,
        ready: true,
        transport: "smtp",
        from_email: "otari@example.com",
        from_name: "Otari",
        public_base_url: "https://gateway.example.com",
        missing: [],
      },
    },
  },
}

/**
 * Configured but incomplete. `missing` is in config order and is empty exactly
 * when `ready` is true, so the card can name what is still needed rather than
 * saying only that mail does not work.
 */
export const MissingSettings: Story = {
  parameters: {
    api: {
      [`${API_ROOT}/settings/mail`]: {
        enabled: true,
        ready: false,
        transport: "smtp",
        from_email: null,
        from_name: "Otari",
        public_base_url: null,
        missing: ["mail.from_email", "mail.public_base_url"],
      },
    },
  },
}

/**
 * No transport at all, which is the default for a self-hosted gateway. Every
 * email-bearing flow (invitations, password recovery) is unavailable until this is
 * set, which is why the e2e screenshot suite stubs the bootstrap to flip it.
 */
export const NotConfigured: Story = {
  parameters: {
    api: {
      [`${API_ROOT}/settings/mail`]: {
        enabled: false,
        ready: false,
        transport: null,
        from_email: null,
        from_name: "Otari",
        public_base_url: null,
        missing: ["mail.transport"],
      },
    },
  },
}

/**
 * The gateway refused. `retry: false` on the catalog's query client means this is
 * the state the card settles in immediately, rather than after three attempts.
 */
export const GatewayError: Story = {
  parameters: {
    api: {
      [`${API_ROOT}/settings/mail`]: {
        $status: 503,
        $body: { detail: "Settings store unavailable." },
      },
    },
  },
}
