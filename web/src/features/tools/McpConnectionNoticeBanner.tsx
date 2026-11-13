import { useEffect, useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { McpNotice } from "@/features/tools/McpNotice"
import {
  forgetMcpConnectionNotice,
  readMcpConnectionNotice,
  subscribeToMcpConnectionNotice,
} from "@/features/tools/mcpConnectionNotice"
import { asksForOAuthClient } from "@/features/tools/mcpErrors"

/**
 * An MCP sign-in that did not finish, said once on the page it came back to: it
 * failed, or was canceled at the provider. One that worked says nothing, since
 * the page shows the result. A sign-in that ran in a popup reports to the page
 * still on screen, so the banner also takes a notice set while it shows.
 * Renders nothing otherwise.
 */
export function McpConnectionNoticeBanner({
  onAddClient,
}: {
  /** Opens Settings on the OAuth client ID, for an admin whose sign-in asked for one. */
  onAddClient?: () => void
} = {}) {
  const [notice, setNotice] = useState(readMcpConnectionNotice)
  useEffect(() => {
    forgetMcpConnectionNotice()
    return subscribeToMcpConnectionNotice((next) => {
      setNotice(next)
      forgetMcpConnectionNotice()
    })
  }, [])
  if (notice === undefined) return null
  return (
    <McpNotice
      tone={notice.tone}
      message={notice.message}
      action={
        onAddClient && asksForOAuthClient(notice.message) ? (
          <Button size="sm" onPress={onAddClient}>
            Open Settings
          </Button>
        ) : undefined
      }
      onDismiss={() => setNotice(undefined)}
    />
  )
}
