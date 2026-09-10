import { Button } from "@heroui/react"
import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { ConfirmDialog } from "./ConfirmDialog"

const meta = {
  title: "Design system/Feedback/ConfirmDialog",
  component: ConfirmDialog,
  args: {
    isOpen: true,
    onOpenChange: () => {},
    heading: "Delete 3 keys?",
    body: "The callers using them will start receiving 401s immediately. This cannot be undone.",
    confirmLabel: "Delete keys",
    isPending: false,
    onConfirm: () => {},
  },
} satisfies Meta<typeof ConfirmDialog>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Controlled and open. This is the dialog for every delete of a record, one row
 * or a selection of them (otari-ai#2110). `ConfirmButton`'s two-click arming is
 * what is left, for a destructive action that deletes nothing.
 */
export const Danger: Story = {}

/** A non-destructive confirmation, for a change that is merely significant. */
export const Primary: Story = {
  args: {
    heading: "Refresh the pricing catalog?",
    body: "Rates are re-fetched for all 42 models. Any manual override you set is kept.",
    confirmLabel: "Refresh pricing",
    confirmVariant: "primary",
  },
}

/** While the mutation is in flight, so a second press cannot fire it twice. */
export const Pending: Story = {
  args: { isPending: true },
}

/**
 * The mutation came back with a refusal. The dialog stays open holding the
 * gateway's own message, rather than closing and losing it.
 */
export const WithError: Story = {
  args: {
    error: new Error("2 of the 3 keys were already revoked."),
  },
}

/** `body` is a node, so it can carry a list of exactly what is about to happen. */
export const RichBody: Story = {
  args: {
    body: (
      <div className="flex flex-col gap-2">
        <p>These keys will stop working immediately:</p>
        <ul className="flex list-disc flex-col gap-1 pl-5 font-mono text-caption">
          <li>ci-pipeline</li>
          <li>staging-app</li>
          <li>notebook</li>
        </ul>
      </div>
    ),
  },
}

/** Driven from a trigger, which is how a page actually opens it. */
export const FromTrigger: Story = {
  render: (args) => {
    const [open, setOpen] = useState(false)
    return (
      <>
        <Button variant="danger" onPress={() => setOpen(true)}>
          Delete selected
        </Button>
        <ConfirmDialog
          {...args}
          isOpen={open}
          onOpenChange={setOpen}
          onConfirm={() => setOpen(false)}
        />
      </>
    )
  },
}
