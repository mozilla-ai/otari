import type { Meta, StoryObj } from "@storybook/react-vite"

import { TableScrollFrame } from "../layout/TableScrollFrame"
import { DataTable, type DataTableColumn } from "./DataTable"

/**
 * TEMPORARY measurement harness. Not a catalog entry: delete with the change
 * that adds it.
 *
 * Renders `DataTable` inside every per-feature wrapper class the stylesheet
 * targets, with exactly the column ids that stylesheet names, so a Playwright
 * pass can read the computed geometry of each lane before and after the
 * component swaps off HeroUI's Table. The contexts and their keys were derived
 * from globals.css rather than transcribed from the feature files, so the
 * harness measures what the CSS actually reaches.
 */
const CONTEXTS: { place: string; keys: string[] }[] = [
  // No wrapper class, which is a real case rather than a control: three feature
  // cards (OrganizationBudgetsCard, SpendCeilingsCard, OrganizationRosterCard)
  // render a bare DataTable, so they take whatever the base `.otari-table`
  // rules say. Measured here so a change to that base shows up as drift instead
  // of arriving unseen on three surfaces.
  { place: "", keys: ["name", "value", "actions"] },
  {
    place: "otari-accounts-table",
    keys: [
      "access",
      "account",
      "actions",
      "last-sign-in",
      "organizations",
      "status",
    ],
  },
  {
    place: "otari-activity-table",
    keys: [
      "api_key",
      "cost",
      "latency",
      "model",
      "routing",
      "status",
      "time",
      "tokens",
      "user",
    ],
  },
  {
    place: "otari-breakdown",
    keys: ["calls", "failed", "requests", "spend", "tokens"],
  },
  {
    place: "otari-budgets-table",
    keys: ["actions", "budget", "default-for", "usage", "users"],
  },
  { place: "otari-domains-table", keys: ["name", "value", "actions"] },
  {
    place: "otari-keys-table",
    keys: [
      "actions",
      "created",
      "expires",
      "key",
      "last_used",
      "owner",
      "status",
    ],
  },
  { place: "otari-mcp-table", keys: ["actions", "enabled", "token"] },
  {
    place: "otari-members-table",
    keys: ["actions", "member", "role", "spend", "status", "workspaces"],
  },
  {
    place: "otari-models-table",
    keys: ["modalities", "model", "policy", "provider"],
  },
  {
    place: "otari-overview-activity",
    keys: ["cost", "key", "model", "status", "time", "tokens"],
  },
  {
    place: "otari-pricing-table",
    keys: [
      "cacheRead",
      "cache_read",
      "cache_write",
      "input",
      "modelKey",
      "model_key",
      "output",
      "spacer",
      "tiers",
      "updatedAt",
    ],
  },
  {
    place: "otari-provider-keys-table",
    keys: ["actions", "api_base", "api_key", "created", "name", "provider"],
  },
  {
    place: "otari-rate-overrides-table",
    keys: [
      "actions",
      "cacheRead",
      "cache_read",
      "cache_write",
      "input",
      "output",
      "period",
      "status",
    ],
  },
  {
    place: "otari-workspaces-table",
    keys: ["actions", "created", "default-budget", "name"],
  },
]

interface DemoRow {
  id: string
  [key: string]: string
}

const ROWS: DemoRow[] = [{ id: "row-a" }, { id: "row-b" }, { id: "row-c" }]

function columnsFor(keys: string[]): DataTableColumn<DemoRow>[] {
  return keys.map((key) => ({
    id: key,
    header: key,
    // Two lines of content, because several tables set their row height from a
    // two-line cell and a single line would not exercise it.
    cell: () => (
      <span className="flex flex-col">
        <span>{key}</span>
        <span className="text-caption">detail</span>
      </span>
    ),
  }))
}

const meta = {
  title: "Zz-measure/TableGeometry",
  parameters: { layout: "fullscreen" },
} satisfies Meta

export default meta

export const AllContexts: StoryObj = {
  render: () => (
    <div className="flex flex-col gap-10 p-4">
      {CONTEXTS.map(({ place, keys }) => (
        <section key={place || "bare"} data-measure={place || "(no wrapper)"}>
          <h2 className="text-overline">{place || "(no wrapper)"}</h2>
          <TableScrollFrame className={place}>
            <DataTable
              ariaLabel={place}
              columns={columnsFor(keys)}
              rows={ROWS}
              getRowKey={(row) => row.id}
              selectionMode="multiple"
              selectedKeys={new Set(["row-b"])}
              onSelectionChange={() => {}}
            />
          </TableScrollFrame>
        </section>
      ))}
    </div>
  ),
}
