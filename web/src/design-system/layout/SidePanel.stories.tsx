import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Button } from "../actions/Button"
import { SidePanel } from "./SidePanel"

/**
 * A record opened over the page from its right edge, with previous and next to
 * walk the list without closing, expand to take the viewport, and close.
 *
 * Which record is open is the caller's: the story keeps an index, as a page
 * keeps the open id in its URL.
 */
const meta = {
  title: "Design system/Layout/SidePanel",
  component: SidePanel,
  args: {
    isOpen: true,
    onClose: () => {},
    label: "Session",
    heading: "Session",
    children: null,
  },
  parameters: { layout: "fullscreen" },
} satisfies Meta<typeof SidePanel>

export default meta

type Story = StoryObj<typeof meta>

const RECORDS = ["session-a1", "session-b2", "session-c3"]

export const WalkingAList: Story = {
  render: () => {
    const [index, setIndex] = useState<number | null>(0)
    return (
      <div className="p-6">
        <Button onPress={() => setIndex(0)}>Open the first record</Button>
        <SidePanel
          isOpen={index !== null}
          onClose={() => setIndex(null)}
          label="Session"
          heading={
            <span className="font-mono text-caption">
              {RECORDS[index ?? 0]}
            </span>
          }
          onPrevious={index ? () => setIndex(index - 1) : undefined}
          onNext={
            index !== null && index < RECORDS.length - 1
              ? () => setIndex(index + 1)
              : undefined
          }
        >
          <p className="p-5 text-body">The open record&apos;s body.</p>
        </SidePanel>
      </div>
    )
  },
}
