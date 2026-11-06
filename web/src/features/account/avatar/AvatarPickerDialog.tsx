import { useRef, useState } from "react"
import { FiShuffle } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { Dialog } from "@/design-system/feedback/Dialog"

import { AvatarField, type ChangeSource, FIELD_FACES } from "./AvatarField"
import { face, sampleFaces } from "./faces"
import { EASE_OUT, prefersReducedMotion } from "./motion"

/**
 * A fresh deal of faces for the field, with `keep` in the slot it had (or the
 * first slot, on opening), so the face in the frame stays under the person's
 * eye while everything around it changes.
 */
function deal(keep: number, previous?: readonly number[]): number[] {
  const faces = sampleFaces(FIELD_FACES).filter((id) => id !== keep)
  const slot = Math.max(0, previous?.indexOf(keep) ?? 0)
  faces.splice(slot, 0, keep)
  return faces.slice(0, FIELD_FACES)
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
  const [order, setOrder] = useState(() => deal(initial))
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
          onPress={() => setOrder((previous) => deal(draft, previous))}
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
