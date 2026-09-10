import type { Meta, StoryObj } from "@storybook/react-vite"

import { TableScrollFrame } from "./TableScrollFrame"

/**
 * The frame a wide table scrolls inside.
 *
 * It exists because of the rule in responsiveness.md that the page body must
 * never scroll horizontally: a table with more columns than a phone has room
 * for has to scroll *within its own frame*, so the shell's chrome stays put and
 * the operator's scroll gesture means one thing.
 *
 * In `layout/` rather than `data/`, which looks arbitrary and is not: layout.md
 * lists it among the band components, because what it frames is a band of the
 * page rather than a property of the table inside it.
 */
const meta = {
  title: "Design system/Layout/TableScrollFrame",
  component: TableScrollFrame,
  args: { className: "", children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof TableScrollFrame>

export default meta

type Story = StoryObj<typeof meta>

/** Narrow the canvas: the header and the rows scroll together, the page does not. */
export const Default: Story = {
  render: () => (
    <div className="w-96">
      <TableScrollFrame className="otari-table">
        <table className="w-max min-w-full">
          <thead>
            <tr>
              {["Key", "Workspace", "Created", "Last used", "Spend"].map(
                (head) => (
                  <th
                    key={head}
                    scope="col"
                    className="whitespace-nowrap px-3 py-2 text-left text-overline"
                  >
                    {head}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {[
              [
                "checkout-service",
                "production",
                "2026-01-04",
                "6m ago",
                "$412.08",
              ],
              ["batch-ingest", "production", "2026-02-11", "2h ago", "$88.40"],
              ["staging-probe", "staging", "2026-03-02", "5d ago", "$1.20"],
            ].map((row) => (
              <tr key={row[0]} className="border-t border-border">
                {row.map((cell) => (
                  <td
                    key={cell}
                    className="whitespace-nowrap px-3 py-2 font-mono text-mono-caption"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </TableScrollFrame>
    </div>
  ),
}
