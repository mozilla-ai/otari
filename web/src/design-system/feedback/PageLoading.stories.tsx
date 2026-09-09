import type { Meta, StoryObj } from "@storybook/react-vite"

import { PageLoading } from "./PageLoading"

const meta = {
  title: "Design system/Feedback/PageLoading",
  component: PageLoading,
} satisfies Meta<typeof PageLoading>

export default meta

type Story = StoryObj<typeof meta>

/**
 * The default label. This exists so a page gated on a first fetch does not flash
 * a bare header over blank space, which reads as broken.
 */
export const Default: Story = {}

export const CustomLabel: Story = {
  args: { label: "Discovering models…" },
}

/** Under a header, which is where it actually appears. */
export const InPage: Story = {
  render: (args) => (
    <div className="flex w-[36rem] flex-col gap-3">
      <h1 className="text-display">Models</h1>
      <PageLoading {...args} />
    </div>
  ),
}
