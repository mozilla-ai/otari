import type { Meta, StoryObj } from "@storybook/react-vite"
import { FiCheck } from "react-icons/fi"

import { Button } from "../actions/Button"
import { CodeBlock } from "../content/CodeBlock"
import { Dialog, DialogSection } from "./Dialog"
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

/** The plain frame: a header, one band to read, and one way out. */
export const Default: Story = {
  args: {
    description: "It lands in Default workspace.",
    children: (
      <DialogSection>
        <p className="text-body">
          Whatever the frame is presenting. A guided step, a receipt, a thing to
          read and copy.
        </p>
      </DialogSection>
    ),
    actions: <Button variant="primary">Done</Button>,
  },
}

/** Several bands, divided edge to edge rather than stacked with gaps. */
export const Divided: Story = {
  args: {
    ...Default.args,
    children: (
      <>
        <DialogSection>
          <p className="text-emphasis">The first thing</p>
          <p className="text-caption text-subtle">
            Each band carries its own padding and the rule above it.
          </p>
        </DialogSection>
        <DialogSection>
          <p className="text-emphasis">The second thing</p>
          <CodeBlock label="curl" value="curl https://example.com" />
        </DialogSection>
      </>
    ),
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
export const Large: Story = {
  args: {
    size: "lg",
    isAnnouncement: true,
    description:
      "Usage, spend and the activity log stay empty until one does, so this guide watches for it and finishes here.",
    children: (
      <>
        <DialogSection>
          <InfoBanner tone="warning">
            Copy this key now. It is shown once.
          </InfoBanner>
        </DialogSection>
        <DialogSection>
          <CodeBlock
            label="curl"
            arrangement="bare"
            value={`curl 'https://gateway.example.com/api/v1/chat/completions' \\\n  -H "Otari-Key: gw-..."`}
          />
        </DialogSection>
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

/** The payoff shape: a mark beside the heading, and a receipt band under it. */
export const WithMark: Story = {
  args: {
    size: "md",
    isAnnouncement: true,
    title: "Your first call went through",
    description:
      "Otari observed the request and finished setup for this workspace.",
    mark: (
      <span className="mt-0.5 flex shrink-0">
        <FiCheck aria-hidden className="text-success size-6" />
      </span>
    ),
    children: (
      <div className="border-border flex border-t">
        <div className="flex min-w-0 flex-1 flex-col gap-1 px-6 py-3">
          <span className="text-mono-overline">Model</span>
          <span className="text-mono-caption text-foreground truncate">
            openai:gpt-4o-mini
          </span>
        </div>
        <div className="border-border flex shrink-0 flex-col gap-1 border-l px-4 py-3 pr-6">
          <span className="text-mono-overline">Latency</span>
          <span className="text-mono-caption text-foreground">412 ms</span>
        </div>
      </div>
    ),
    actions: <Button variant="primary">Continue to the activity log</Button>,
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

/** The two narrower steps, for reference. */
export const Small: Story = { args: { ...Default.args, size: "sm" } }
export const Medium: Story = { args: { ...Default.args, size: "md" } }
