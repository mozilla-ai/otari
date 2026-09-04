export function getModalBackdrop(): HTMLElement {
  const backdrop = document.querySelector<HTMLElement>(
    '[data-slot="modal-backdrop"]',
  )
  if (backdrop === null) {
    throw new Error("Expected an open modal backdrop")
  }
  return backdrop
}
