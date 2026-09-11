import type { Meta, StoryObj } from "@storybook/react-vite"

import { Button } from "../actions/Button"
import { ScanBorder } from "./ScanBorder"

function Row({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex w-full flex-wrap items-center justify-between gap-3">
      <div className="flex flex-col gap-0.5">
        <span className="text-body">{title}</span>
        <span className="text-caption">{detail}</span>
      </div>
      <Button variant="ghost" size="sm">
        Check now
      </Button>
    </div>
  )
}

const meta = {
  title: "Design system/Feedback/ScanBorder",
  component: ScanBorder,
  args: {
    isActive: true,
    className: "bg-surface-alt max-w-lg px-4 py-3",
    children: (
      <Row
        title="Listening for your first request"
        detail="This sheet notices it within a few seconds."
      />
    ),
  },
} satisfies Meta<typeof ScanBorder>

export default meta

type Story = StoryObj<typeof meta>

/** Waiting: the arc travels the edge over the resting hairline. */
export const Waiting: Story = {}

/** Nothing left to wait for, so the edge is a plain hairline again. */
export const Resting: Story = { args: { isActive: false } }

/** A failure keeps the wait and turns the arc red. */
export const Failed: Story = {
  args: {
    tone: "danger",
    className: "bg-danger-subtle max-w-lg px-4 py-3",
    children: (
      <Row
        title="Request failed: a budget rejected the request."
        detail="Still listening. Fix it and send the request again."
      />
    ),
  },
}
