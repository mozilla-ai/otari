import { useRef, useState } from "react"
import { FiShuffle } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { Dialog } from "@/design-system/feedback/Dialog"

import { AvatarField, type ChangeSource } from "./AvatarField"
import { FACE_COUNT, face } from "./faces"
import { EASE_OUT, prefersReducedMotion } from "./motion"

function shuffled(): number[] {
  const order = Array.from({ length: FACE_COUNT }, (_, i) => i)
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return order
}

/**
 * A fresh order with `keep` left in the slot it had, so the selected face stays
 * under the person's eye while everything around it rearranges.
 */
function reshuffled(previous: readonly number[], keep: number): number[] {
  const order = shuffled()
  const slot = previous.indexOf(keep)
  const from = order.indexOf(keep)
  ;[order[slot], order[from]] = [order[from], order[slot]]
  return order
}

/**
 * Pick a face. Nothing is saved until Save: dragging around and trying faces
 * is free, and Cancel puts back what you had.
 */
export function AvatarPickerDialog({
  isOpen,
  onOpenChange,
  initial,
  onSave,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  initial: number
  onSave: (id: number) => void
}) {
  const [draft, setDraft] = useState(initial)
  const [order, setOrder] = useState(shuffled)
  const source = useRef<ChangeSource | null>(null)

  return (
    <Dialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      title="Choose an avatar"
      description="Shown beside your name in the sidebar. Drag to see more."
      footerStart={
        <Button
          size="sm"
          onPress={() => setOrder((previous) => reshuffled(previous, draft))}
        >
          <FiShuffle aria-hidden className="h-3.5 w-3.5" />
          Shuffle
        </Button>
      }
      actions={
        <>
          <Button size="sm" onPress={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            variant="primary"
            onPress={() => {
              onSave(draft)
              onOpenChange(false)
            }}
          >
            Save
          </Button>
        </>
      }
    >
      <div className="border-border flex min-h-0 flex-1 flex-col border-t">
        <AvatarField
          order={order}
          value={draft}
          onChange={(id, from) => {
            source.current = from
            setDraft(id)
          }}
        />
      </div>
      <p className="border-border text-muted border-t px-6 py-3 text-sm capitalize">
        {/* The name crossfades through a 2px blur, so the old word and the
            new one read as one word changing rather than two overlapping.
            Not on a key press, and not on the first paint. */}
        <span
          key={draft}
          className="inline-block"
          ref={(node) => {
            if (!node || !source.current || source.current === "key") return
            if (prefersReducedMotion()) return
            source.current = null
            node.animate(
              [
                {
                  opacity: 0,
                  filter: "blur(2px)",
                  transform: "translateY(3px)",
                },
                { opacity: 1, filter: "blur(0)", transform: "translateY(0)" },
              ],
              { duration: 180, easing: EASE_OUT },
            )
          }}
        >
          {face(draft).label}
        </span>
      </p>
    </Dialog>
  )
}
