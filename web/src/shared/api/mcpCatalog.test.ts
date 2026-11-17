import { describe, expect, it } from "vitest"

import { mcpCatalogSyncRefetchMs } from "@/shared/api/mcpCatalog"
import { MCP_CATALOG_SYNC_POLL_MS } from "@/shared/api/queryKeys"
import { mcpCatalogSyncStatus } from "@/tests/fixtures"

const NOW = Date.parse("2026-11-13T10:00:00Z")

describe("mcpCatalogSyncRefetchMs", () => {
  it("polls while a sync runs or waits for a worker", () => {
    expect(
      mcpCatalogSyncRefetchMs(mcpCatalogSyncStatus({ running: true }), NOW),
    ).toBe(MCP_CATALOG_SYNC_POLL_MS)
    expect(
      mcpCatalogSyncRefetchMs(mcpCatalogSyncStatus({ requested: true }), NOW),
    ).toBe(MCP_CATALOG_SYNC_POLL_MS)
  })

  it("reads the status again just after a cooldown ends", () => {
    const status = mcpCatalogSyncStatus({
      next_request_at: "2026-11-13T10:05:00Z",
    })
    expect(mcpCatalogSyncRefetchMs(status, NOW)).toBe(5 * 60_000 + 1_000)
  })

  it("sets no timer when a sync may be asked for now", () => {
    expect(mcpCatalogSyncRefetchMs(mcpCatalogSyncStatus(), NOW)).toBe(false)
    expect(mcpCatalogSyncRefetchMs(undefined, NOW)).toBe(false)
  })
})
