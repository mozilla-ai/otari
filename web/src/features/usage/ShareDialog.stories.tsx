import type { Meta, StoryObj } from "@storybook/react-vite"

import type { UsageGroupRow } from "@/client"
import { seriesPoint, usageTotals } from "@/tests/fixtures"

import { ShareDialog } from "./ShareDialog"

/**
 * The composer around `ShareCard`: pick a ratio, a hero stat and how many model
 * rows to print, then export the PNG.
 *
 * Entirely prop-driven, it takes the page's own query results, so it never
 * refetches and cannot disagree with the numbers the operator was just looking at.
 *
 * The export itself is the part a catalog cannot prove. It rasterizes through
 * canvas, `toBlob` and an object URL, none of which jsdom implements (which is why
 * `src/tests/setup.ts` stubs all three), so pressing download here exercises the
 * real path in a way neither the unit suite nor this catalog asserts.
 */
const DAYS = Array.from({ length: 30 }, (_, index) =>
  new Date(Date.UTC(2026, 6, 27 + index)).toISOString(),
)

const SERIES = DAYS.map((bucket, index) =>
  seriesPoint({
    bucket_start: bucket,
    requests: 480 + Math.round(Math.sin(index / 4) * 180) + index * 12,
    cost: 9.4 + Math.cos(index / 5) * 3.1,
    tokens: 118_000 + index * 2_400,
  }),
)

const MODEL_ROWS: UsageGroupRow[] = [
  {
    key: "anthropic:claude-haiku-4-5",
    label: "claude-haiku-4-5",
    requests: 11_204,
    tokens: 2_400_000,
    cost: 118.2,
    is_other: false,
  },
  {
    key: "openai:gpt-4o-mini",
    label: "gpt-4o-mini",
    requests: 5_812,
    tokens: 1_180_000,
    cost: 84.6,
    is_other: false,
  },
  {
    key: "openai:gpt-4o",
    label: "gpt-4o",
    requests: 1_204,
    tokens: 640_000,
    cost: 186.4,
    is_other: false,
  },
  {
    key: null,
    label: "other",
    requests: 182,
    tokens: 61_000,
    cost: 23.7,
    is_other: true,
  },
]

const meta = {
  title: "Dashboard/Usage/ShareDialog",
  component: ShareDialog,
  args: {
    totals: usageTotals({
      request_count: 18_402,
      cost: 412.9,
      total_tokens: 4_100_000,
    }),
    series: SERIES,
    modelRows: MODEL_ROWS,
    windowLabel: "Last 30 days",
    scopeSuffix: "",
    startIso: DAYS[0],
    endIso: DAYS[DAYS.length - 1],
    isStale: false,
    onClose: () => {},
  },
  parameters: { layout: "fullscreen" },
} satisfies Meta<typeof ShareDialog>

export default meta

type Story = StoryObj<typeof meta>

/** The whole window, unfiltered. */
export const Default: Story = {}

/**
 * With entity filters applied. `scopeSuffix` is what keeps the card's denominator
 * unambiguous: a card saying "18,402 requests" means something different if it was
 * two models out of forty.
 */
export const FilteredScope: Story = {
  args: {
    scopeSuffix: "2 models, 1 key",
    totals: usageTotals({
      request_count: 6_204,
      cost: 96.4,
      total_tokens: 1_240_000,
    }),
    modelRows: MODEL_ROWS.slice(0, 2),
  },
}

/** A single day, which gives the card very little to rank. */
export const ShortWindow: Story = {
  args: {
    windowLabel: "Last 24 hours",
    startIso: DAYS[DAYS.length - 2],
    endIso: DAYS[DAYS.length - 1],
    totals: usageTotals({
      request_count: 612,
      cost: 14.2,
      total_tokens: 138_000,
    }),
    modelRows: MODEL_ROWS.slice(0, 1),
    series: SERIES.slice(-2),
  },
}

/**
 * While the page's own query is in flight. The dialog says so rather than
 * exporting numbers that are about to change.
 */
export const Stale: Story = {
  args: { isStale: true },
}

/**
 * A gateway serving only self-hosted models: no cost anywhere, so the cost stat is
 * omitted rather than published as a proud "$0.00".
 */
export const NoCost: Story = {
  args: {
    totals: usageTotals({
      request_count: 18_402,
      cost: 0,
      total_tokens: 4_100_000,
    }),
    modelRows: MODEL_ROWS.map((row) => ({ ...row, cost: 0 })),
  },
}

/** Nothing served yet. */
export const NoData: Story = {
  args: {
    totals: usageTotals(),
    series: [],
    modelRows: [],
    windowLabel: "Last 30 days",
  },
}
