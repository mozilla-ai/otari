import type { Meta, StoryObj } from "@storybook/react-vite"

import { Section } from "./Section"

/**
 * A band of the page: rules that run the full width of the scroll area, with
 * the content inside them still in the centered column.
 *
 * Two elements, because one cannot be both full-width and centered.
 *
 * **Every story here passes `bleed={false}`**, and that is the interesting part
 * rather than a workaround. The escape is `100cqw` measured against `<main>`,
 * so a bleeding band needs the app's scroll container to escape *to*; in a story
 * canvas there is none, and a bleeding band would overflow the frame instead of
 * stopping at a column. It is the same reason a band nested inside a narrow cell
 * has to say `bleed={false}` in the real app.
 */
const meta = {
  title: "Design system/Layout/Section",
  component: Section,
  args: { bleed: false, children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Section>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <Section bleed={false} className="border-y border-border py-4">
      <p className="text-body">
        A band's content, in the column the band restores.
      </p>
    </Section>
  ),
}

/**
 * Stacked, which is what a page is: the rules between them are each band's own
 * edge, so two adjacent bands share one line rather than drawing two.
 */
export const Stacked: Story = {
  render: () => (
    <div className="flex flex-col">
      {["Totals", "Breakdown", "Recent activity"].map((title) => (
        <Section
          key={title}
          bleed={false}
          className="border-t border-border py-4"
        >
          <span className="text-heading">{title}</span>
        </Section>
      ))}
    </div>
  ),
}

/**
 * `className` styles the band (its rules, its vertical padding, its own layout
 * when the content is one row); `contentClassName` styles the column inside it.
 * Mixing them up is the mistake this pair exists to make visible: a fill on the
 * content stops at the column's edge, where a fill on the band spans the page.
 */
export const BandVersusContent: Story = {
  render: () => (
    <div className="flex flex-col gap-6">
      <Section
        bleed={false}
        className="border-y border-border bg-surface-alt py-4"
      >
        <span className="text-caption">className: the fill spans the band</span>
      </Section>
      <Section
        bleed={false}
        className="border-y border-border py-4"
        contentClassName="bg-surface-alt p-3"
      >
        <span className="text-caption">
          contentClassName: the fill stops at the column
        </span>
      </Section>
    </div>
  ),
}
