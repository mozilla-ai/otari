import { FiPlus } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { Tooltip } from "@/design-system/overlays/Tooltip"

/**
 * The rail's Create API key. Collapsed it is a bare "+", named on hover as the
 * rail's other rows are; react-aria's button drops a `title`, so a tooltip
 * says it.
 */
export function CreateKeyRailButton({
  isCollapsed,
  isDense = false,
  onPress,
}: {
  isCollapsed: boolean
  /** The dense sidebar's 32px button in place of the default 36px. */
  isDense?: boolean
  onPress: () => void
}) {
  const button = (
    <Button
      variant="primary"
      // 32px in the dense sidebar; unset, the default 36px.
      size={isDense ? "sm" : undefined}
      className="w-full shrink-0"
      aria-label={isCollapsed ? "Create API key" : undefined}
      onPress={onPress}
    >
      <FiPlus aria-hidden className="size-4" />
      {isCollapsed ? null : "Create API key"}
    </Button>
  )
  return isCollapsed ? (
    <Tooltip content="Create API key" placement="right">
      {button}
    </Tooltip>
  ) : (
    button
  )
}
