import { useLocation } from "@tanstack/react-router"
import {
  lazy,
  type ReactNode,
  type RefObject,
  Suspense,
  useEffect,
  useRef,
  useState,
} from "react"
import {
  type CreateKeyEntryPoint,
  OpenCreateKeyContext,
} from "@/features/keys/openCreateKey"
import { useCanCreateKey } from "@/features/organization/access"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

const loadCreateKeyDialog = () => import("@/features/keys/CreateKeyDialog")
const CreateKeyDialog = lazy(() =>
  loadCreateKeyDialog().then((module) => ({ default: module.CreateKeyDialog })),
)

/**
 * The one Create API key dialog, opened from the rail and from any page
 * through `OpenCreateKeyContext`. The dialog is its own chunk, mounted on the
 * first open.
 */
export function CreateKeyHost({ children }: { children: ReactNode }) {
  const canCreateKey = useCanCreateKey()
  const telemetry = useTelemetry()
  const { pathname } = useLocation()
  const [isOpen, setOpen] = useState(false)
  const [hasOpened, setOpened] = useState(false)
  // Where focus goes on close when the control that opened it is gone, as
  // the opening named it; each opening names its own.
  const namedFocusRef = useRef<HTMLElement | null>(null)
  // The control that was pressed to open it. The key-created step is a second
  // dialog that mounts while focus is still inside the first, so react-aria
  // has no trigger to give focus back to when it closes.
  const openerRef = useRef<HTMLElement | null>(null)
  const [returnFocusRef] = useState<RefObject<HTMLElement | null>>(() => ({
    get current() {
      return namedFocusRef.current ?? openerRef.current
    },
    set current(_) {},
  }))
  // A key's secret is on screen, shown once.
  const [isSecretShown, setSecretShown] = useState(false)
  const open = (
    entryPoint: CreateKeyEntryPoint,
    returnFocusTo: HTMLElement | null = null,
  ) => {
    if (!isOpen && document.activeElement instanceof HTMLElement) {
      openerRef.current = document.activeElement
    }
    if (!isOpen) namedFocusRef.current = returnFocusTo
    // Only a closed dialog opens: a press that lands on an open one is not
    // another opening.
    if (!isOpen) {
      telemetry.recordEvent(TELEMETRY_EVENTS.CREATE_API_KEY_OPENED, {
        entry_point: entryPoint,
      })
    }
    setOpened(true)
    setOpen(true)
  }
  // Fetched once the shell is idle, so the first open does not wait on it.
  useEffect(() => {
    if (!canCreateKey) return
    const id = window.setTimeout(() => void loadCreateKeyDialog(), 2_000)
    return () => window.clearTimeout(id)
  }, [canCreateKey])
  // The dialog belongs to the page it was opened on: Back or Forward under it
  // closes it rather than leaving it over a page it was not opened from.
  const [openedAt, setOpenedAt] = useState<string>()
  if (isOpen && openedAt === undefined) {
    setOpenedAt(pathname)
  } else if (isOpen && openedAt !== pathname) {
    // An unseen secret outlives a navigation: Back would otherwise lose a key
    // that exists. It stays over the page it landed on until it is
    // acknowledged.
    if (isSecretShown) {
      setOpenedAt(pathname)
    } else {
      setOpen(false)
      setOpenedAt(undefined)
    }
  } else if (!isOpen && openedAt !== undefined) {
    setOpenedAt(undefined)
  }

  return (
    <OpenCreateKeyContext value={open}>
      {children}
      {hasOpened ? (
        <Suspense fallback={null}>
          <CreateKeyDialog
            isOpen={isOpen}
            onOpenChange={setOpen}
            returnFocusRef={returnFocusRef}
            onSecretShownChange={setSecretShown}
          />
        </Suspense>
      ) : null}
    </OpenCreateKeyContext>
  )
}
