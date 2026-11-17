import { Dropdown, Label } from "@heroui/react"
import { FiChevronRight, FiMoreHorizontal } from "react-icons/fi"

import type { HostedVendor } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { Meter } from "@/design-system/metrics/Meter"

/** What a vendor's menu does to all of its models at once. */
export type VendorBulkAction =
  | "serve-all"
  | "withhold-all"
  | "withhold-unpriced"
  | "reset"

const BULK_ACTIONS: { id: VendorBulkAction; label: string }[] = [
  { id: "serve-all", label: "Serve all models" },
  { id: "withhold-all", label: "Withhold all models" },
  { id: "withhold-unpriced", label: "Withhold unpriced models" },
  { id: "reset", label: "Reset custom prices to default…" },
]

/**
 * One vendor of a hosted provider: what it serves of its models, its own
 * switch, and the actions that set every model of it at once.
 *
 * The switch withholds the whole vendor and leaves each model's switch as it
 * is, so it is a separate control from the bulk menu, which writes those
 * per-model switches.
 */
export function HostedVendorRow({
  vendor,
  isOpen,
  isSwitching,
  isBulkPending,
  onToggleOpen,
  onSwitch,
  onBulk,
}: {
  vendor: HostedVendor
  isOpen: boolean
  /** This vendor's switch is being written. */
  isSwitching: boolean
  /** A bulk action is in flight, on this vendor or another. */
  isBulkPending: boolean
  onToggleOpen: () => void
  onSwitch: (enabled: boolean) => void
  onBulk: (action: VendorBulkAction) => void
}) {
  const name = vendor.vendor
  return (
    <div className="flex min-h-11 flex-wrap items-center gap-x-3 gap-y-1 py-1 pr-2 pl-1">
      <Button
        isIconOnly
        size="sm"
        aria-label={`${isOpen ? "Hide" : "Show"} ${name} models`}
        aria-expanded={isOpen}
        onPress={onToggleOpen}
      >
        <FiChevronRight
          aria-hidden
          className={`size-4 text-subtle transition-transform motion-reduce:transition-none ${
            isOpen ? "rotate-90" : ""
          }`}
        />
      </Button>
      <span className="flex min-w-0 flex-1 flex-wrap items-center gap-x-2 gap-y-1">
        <span className={vendor.enabled ? "text-emphasis" : "text-subtle"}>
          {name}
        </span>
        {vendor.enabled ? null : <Badge tone="warn">Vendor off</Badge>}
        {/* A switch stored under a name no model goes by any more, kept in
            view so it can be switched back on. */}
        {vendor.total === 0 ? <Badge tone="muted">No models</Badge> : null}
        {vendor.unpriced > 0 ? (
          <Badge tone="warn">{`${vendor.unpriced} unpriced`}</Badge>
        ) : null}
        {vendor.unavailable > 0 ? (
          <Badge tone="warn">{`${vendor.unavailable} unavailable`}</Badge>
        ) : null}
        {vendor.custom_differs > 0 ? (
          <Badge tone="muted">{`${vendor.custom_differs} custom ≠ default`}</Badge>
        ) : null}
      </span>
      <span className="flex shrink-0 items-center gap-2 text-caption text-subtle tabular-nums">
        <span className="max-md:hidden">
          <Meter
            fraction={vendor.total === 0 ? 0 : vendor.serving / vendor.total}
            ariaLabel={`${vendor.serving} of ${vendor.total} ${name} models serving`}
          />
        </span>
        {vendor.serving} of {vendor.total} serving
      </span>
      <span className="flex shrink-0 items-center gap-1">
        <Toggle
          isSelected={vendor.enabled}
          isDisabled={isSwitching}
          label={`Serve ${name} models`}
          onChange={onSwitch}
        />
        <Dropdown>
          <Dropdown.Trigger
            aria-label={`More for ${name}`}
            isDisabled={isBulkPending || vendor.total === 0}
            className="inline-flex size-11 items-center justify-center text-muted hover:text-foreground"
          >
            <FiMoreHorizontal aria-hidden className="size-4 shrink-0" />
          </Dropdown.Trigger>
          <Dropdown.Popover placement="bottom end">
            <Dropdown.Menu
              aria-label={`Actions for ${name}`}
              onAction={(id) => {
                const action = BULK_ACTIONS.find((item) => item.id === id)
                if (action) onBulk(action.id)
              }}
            >
              {BULK_ACTIONS.map((action) => (
                <Dropdown.Item
                  key={action.id}
                  id={action.id}
                  textValue={action.label}
                >
                  <Label className="text-body">{action.label}</Label>
                </Dropdown.Item>
              ))}
            </Dropdown.Menu>
          </Dropdown.Popover>
        </Dropdown>
      </span>
    </div>
  )
}
