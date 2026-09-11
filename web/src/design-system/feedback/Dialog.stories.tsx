import type { Meta, StoryObj } from "@storybook/react-vite"
import { FiCheck } from "react-icons/fi"

import { Button } from "../actions/Button"
import { CodeBlock } from "../content/CodeBlock"
import { Dialog } from "./Dialog"
import { InfoBanner } from "./InfoBanner"

const meta = {
  title: "Design system/Feedback/Dialog",
  component: Dialog,
  parameters: { layout: "fullscreen" },
  args: {
    isOpen: true,
    onOpenChange: () => {},
    title: "Send your first request",
    children: null,
  },
} satisfies Meta<typeof Dialog>

export default meta

type Story = StoryObj<typeof meta>

/** The plain frame: a header, a body to read, and one way out. */
export const Default: Story = {
  args: {
    description: "It lands in Default workspace.",
    children: (
      <p className="text-body">
        Whatever the frame is presenting. A guided step, a receipt, a thing to
        read and copy.
      </p>
    ),
    actions: <Button variant="primary">Done</Button>,
  },
}

/** With a footer caption beside the controls. */
export const WithFooterCaption: Story = {
  args: {
    ...Default.args,
    footerStart: (
      <p className="text-caption">Usage stays empty until a request lands.</p>
    ),
    actions: (
      <>
        <Button>Skip</Button>
        <Button variant="primary">Done</Button>
      </>
    ),
  },
}

/** The widest step, for a frame carrying a key and a runnable example. */
export const ExtraLarge: Story = {
  args: {
    size: "xl",
    isAnnouncement: true,
    description:
      "Usage, spend and the activity log stay empty until one does, so this guide watches for it and finishes here.",
    children: (
      <>
        <InfoBanner tone="warning">
          Copy this key now. It is shown once.
        </InfoBanner>
        <CodeBlock
          label="curl"
          value={`curl 'https://gateway.example.com/api/v1/chat/completions' \\\n  -H "Otari-Key: gw-..."`}
        />
      </>
    ),
    status: (
      <div className="border-border bg-surface-alt flex items-center justify-between gap-3 border px-4 py-3">
        <span className="text-body">Listening for your first request</span>
        <Button variant="ghost" size="sm">
          Check now
        </Button>
      </div>
    ),
    footerStart: <p className="text-caption">Skipping keeps the key.</p>,
    actions: <Button variant="ghost">Skip this guide</Button>,
  },
}

/** The payoff shape: a mark, a heading and a receipt down the middle. */
export const Centered: Story = {
  args: {
    size: "sm",
    align: "center",
    isAnnouncement: true,
    title: "Your first request went through",
    description:
      "This workspace is serving traffic. Usage, spend and the activity log fill in from here.",
    mark: (
      <span className="bg-success-subtle flex size-12 items-center justify-center">
        <FiCheck aria-hidden className="text-success size-6" />
      </span>
    ),
    children: (
      <p className="text-caption text-center font-mono">
        openai:gpt-4o-mini · 412 ms · $0.000123
      </p>
    ),
    actions: (
      <>
        <Button>Dismiss</Button>
        <Button variant="primary">Open the activity log</Button>
      </>
    ),
  },
}

/**
 * Undismissable: no close control and no backdrop press, for a frame whose
 * content cannot be recovered once it goes away.
 */
export const Undismissable: Story = {
  args: {
    ...Default.args,
    isDismissable: false,
    actions: <Button variant="primary">I have copied it</Button>,
  },
}

/** The three narrower steps, for reference. */
export const Small: Story = { args: { ...Default.args, size: "sm" } }
export const Medium: Story = { args: { ...Default.args, size: "md" } }
export const Large: Story = { args: { ...Default.args, size: "lg" } }
