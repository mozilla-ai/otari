import { Toggle } from "@/design-system/forms/Toggle"

/**
 * Whether the Playground asks the gateway to keep each prompt on the session's
 * trace. It is kept only where the workspace keeps content too, and only its
 * sender reads it there.
 */
export function PromptCaptureToggle({
  isSelected,
  onChange,
}: {
  isSelected: boolean
  onChange: (next: boolean) => void
}) {
  return (
    <div className="flex min-h-11 items-center gap-2 md:min-h-9">
      <Toggle
        isSelected={isSelected}
        onChange={onChange}
        label="Capture prompts"
      />
      <span aria-hidden className="text-body">
        Capture prompts
      </span>
    </div>
  )
}
