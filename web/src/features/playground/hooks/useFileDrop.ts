import type { DragEvent } from "react"
import { useRef, useState } from "react"

function carriesFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer.types).includes("Files")
}

/**
 * Accept files dropped anywhere on an element, and say while some are held over it.
 *
 * Counted rather than a boolean, because the pointer crossing into a child
 * fires a leave on the parent before the enter on the child, and a flag would
 * flicker off for every element the drag passes over.
 */
export function useFileDrop(params: {
  isEnabled: boolean
  onDrop: (files: File[]) => void
}) {
  const depth = useRef(0)
  const [isDragging, setIsDragging] = useState(false)

  if (!params.isEnabled) {
    return { isDragging: false, dropProps: {} }
  }

  return {
    isDragging,
    dropProps: {
      onDragEnter: (event: DragEvent) => {
        if (!carriesFiles(event)) return
        event.preventDefault()
        depth.current += 1
        setIsDragging(true)
      },
      onDragOver: (event: DragEvent) => {
        // Without this the browser opens the file instead of dropping it here.
        if (carriesFiles(event)) event.preventDefault()
      },
      onDragLeave: (event: DragEvent) => {
        if (!carriesFiles(event)) return
        depth.current = Math.max(depth.current - 1, 0)
        if (depth.current === 0) setIsDragging(false)
      },
      onDrop: (event: DragEvent) => {
        if (!carriesFiles(event)) return
        event.preventDefault()
        depth.current = 0
        setIsDragging(false)
        params.onDrop(Array.from(event.dataTransfer.files))
      },
    },
  }
}
