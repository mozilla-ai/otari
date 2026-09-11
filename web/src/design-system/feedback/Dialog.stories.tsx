import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Button } from "../actions/Button"
import { Field } from "../forms/Field"
import { Dialog } from "./Dialog"

/**
 * The shell every dialog in this product sits in.
 *
 * It exists because the nest does not fit in a head: `Backdrop` > `Container` >
 * `Dialog` > `Header` > `Heading`, then a `Body` and a `Footer`, six levels
 * before a call site says anything of its own. Five components had written it
 * out by hand, and the copies had begun to differ in the parts that are easy to
 * get silently wrong: placement, dismiss behavior, and which element carries
 * the gap.
 *
 * `ConfirmDialog` is the specialization for a destructive action, with the two
 * buttons and the error line built in. This one is for a dialog holding a form.
 *
 * Note the `layout: "fullscreen"` parameter: a dialog portals out of
 * `#storybook-root`, so the centered layout would measure an empty trigger.
 */
// Required props on the meta, so a story that supplies its own `render` still
// satisfies the component's contract without restating them. Every story here
// renders, because a dialog needs a trigger and its own open state to be worth
// looking at.
const meta = {
  title: "Design system/Feedback/Dialog",
  component: Dialog,
  args: {
    isOpen: false,
    onOpenChange: () => {},
    heading: "Create an API key",
    children: null,
  },
  parameters: { layout: "fullscreen" },
} satisfies Meta<typeof Dialog>

export default meta

type Story = StoryObj<typeof meta>

/** Press the button to open it. Escape and a click outside both close it. */
export const Default: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false)
    const [name, setName] = useState("")
    return (
      <div className="p-6">
        <Button variant="primary" onPress={() => setIsOpen(true)}>
          Create key
        </Button>
        <Dialog
          isOpen={isOpen}
          onOpenChange={setIsOpen}
          heading="Create an API key"
          footer={
            <>
              <Button onPress={() => setIsOpen(false)}>Cancel</Button>
              <Button variant="primary" onPress={() => setIsOpen(false)}>
                Create
              </Button>
            </>
          }
        >
          <Field
            label="Key name"
            value={name}
            onChange={setName}
            placeholder="checkout-service"
            description="Lowercase, hyphens, no spaces."
            reserveMessage
          />
        </Dialog>
      </div>
    )
  },
}

/**
 * The three container sizes. `md` is the default; `lg` is for a dialog holding
 * a table or a code block, `sm` for one asking a single question.
 */
export const Sizes: Story = {
  render: () => {
    const [size, setSize] = useState<"sm" | "md" | "lg" | undefined>(undefined)
    return (
      <div className="flex gap-3 p-6">
        {(["sm", "md", "lg"] as const).map((each) => (
          <Button key={each} onPress={() => setSize(each)}>
            Open {each}
          </Button>
        ))}
        <Dialog
          isOpen={size !== undefined}
          onOpenChange={() => setSize(undefined)}
          heading={`A ${size ?? "md"} dialog`}
          size={size ?? "md"}
          footer={<Button onPress={() => setSize(undefined)}>Close</Button>}
        >
          <p className="text-body">
            The container's width changes; the body's own 4px gap does not.
          </p>
        </Dialog>
      </div>
    )
  },
}

/**
 * `isDismissable={false}` takes Escape and the outside click away, so the
 * footer is the only way out. The only honest reason is unsaved work that would
 * be lost, and even then the better fix is usually to keep the dismiss and
 * confirm the discard.
 */
export const NotDismissable: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false)
    return (
      <div className="p-6">
        <Button onPress={() => setIsOpen(true)}>Open</Button>
        <Dialog
          isOpen={isOpen}
          onOpenChange={setIsOpen}
          heading="Finish setting up this provider"
          isDismissable={false}
          footer={
            <Button variant="primary" onPress={() => setIsOpen(false)}>
              Done
            </Button>
          }
        >
          <p className="text-body">
            Escape does nothing here, and neither does a click on the backdrop.
          </p>
        </Dialog>
      </div>
    )
  },
}

/** Without a footer, for a dialog that only shows something. */
export const NoFooter: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false)
    return (
      <div className="p-6">
        <Button onPress={() => setIsOpen(true)}>Show the request</Button>
        <Dialog
          isOpen={isOpen}
          onOpenChange={setIsOpen}
          heading="Request 4f8a2c9e"
          size="lg"
        >
          <pre className="overflow-x-auto font-mono text-mono-caption">
            {'{\n  "model": "gpt-4o-mini",\n  "stream": true\n}'}
          </pre>
        </Dialog>
      </div>
    )
  },
}
