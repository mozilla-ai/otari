import { Button, Modal } from "@heroui/react"
import { useEffect, useMemo, useRef, useState } from "react"
import type { UsageGroupRow, UsageSeriesPoint, UsageTotals } from "@/client"
import {
  canCopyImages,
  copyBlobAsImage,
  downloadBlob,
  rasterize,
  shareFilename,
} from "@/features/usage/shareImage"
import { ErrorBanner, InfoBanner } from "@/shared/components/ui"
import { CARD_SIZES, type CardRatio, ShareCard } from "./ShareCard"
import {
  availableStats,
  cardModels,
  heroCandidates,
  resolveHero,
  type StatId,
} from "./shareCardData"

// Presentation choices persist so a returning user does not re-pick them. The data
// scope deliberately does NOT: the page's current filters are the user's current
// question, and restoring a stale window from storage would silently change what
// the card claims. That is why there is no window or filter control in here.
const STORE_KEY = "otari.share.presentation.v1"

// The value sets a stored presentation is validated against on read.
const HERO_IDS = [
  "cost",
  "requests",
  "tokens",
  "latency",
] as const satisfies readonly StatId[]
const RATIOS = ["square", "landscape"] as const satisfies readonly CardRatio[]
const THEMES = ["dark", "light"] as const
export const ROW_CHOICES = [1, 3, 5, 9] as const

interface Presentation {
  hero: StatId
  ratio: CardRatio
  theme: "dark" | "light"
  hideDollars: boolean
  rows: number
  title: string
}

const DEFAULTS: Presentation = {
  hero: "requests",
  ratio: "square",
  theme: "dark",
  hideDollars: false,
  rows: 5,
  title: "Where my tokens went",
}

function loadPresentation(): Presentation {
  try {
    const raw = localStorage.getItem(STORE_KEY)
    if (raw === null) {
      return DEFAULTS
    }
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed !== "object" || parsed === null) {
      return DEFAULTS
    }
    // Validated field by field, not spread. A spread copies unknown values
    // straight through, and a stored `ratio` this build no longer has reaches
    // CARD_SIZES[ratio] and throws on the destructure. Anything unrecognized
    // falls back to its default rather than taking the whole object down.
    const stored = parsed as Record<string, unknown>
    const pick = <T,>(value: unknown, allowed: readonly T[], fallback: T): T =>
      allowed.includes(value as T) ? (value as T) : fallback
    return {
      hero: pick(stored.hero, HERO_IDS, DEFAULTS.hero),
      ratio: pick(stored.ratio, RATIOS, DEFAULTS.ratio),
      theme: pick(stored.theme, THEMES, DEFAULTS.theme),
      rows: pick(stored.rows, ROW_CHOICES, DEFAULTS.rows),
      hideDollars:
        typeof stored.hideDollars === "boolean"
          ? stored.hideDollars
          : DEFAULTS.hideDollars,
      title:
        typeof stored.title === "string"
          ? stored.title.slice(0, TITLE_MAX)
          : DEFAULTS.title,
    }
  } catch {
    return DEFAULTS
  }
}

const TITLE_MAX = 60

export interface ShareDialogProps {
  totals: UsageTotals | undefined
  series: UsageSeriesPoint[]
  /** `by_model` rows for the page's current filters, straight off the page's own query. */
  modelRows: UsageGroupRow[]
  windowLabel: string
  /** Names any active entity filters, so the card's denominator is unambiguous. */
  scopeSuffix: string
  startIso: string
  endIso: string
  /** True while the page's own query is in flight or showing placeholder data. */
  isStale: boolean
  onClose: () => void
}

export function ShareDialog(props: ShareDialogProps) {
  const {
    totals,
    series,
    modelRows,
    windowLabel,
    scopeSuffix,
    startIso,
    endIso,
    isStale,
    onClose,
  } = props
  const [presentation, setPresentation] =
    useState<Presentation>(loadPresentation)
  const [preview, setPreview] = useState<string | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<Error | undefined>(undefined)
  const [notice, setNotice] = useState<string | undefined>(undefined)
  const cardRef = useRef<HTMLDivElement>(null)
  // Cleared on unmount: closing the dialog inside the 2s window otherwise fired a
  // state update against an unmounted component.
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )

  const set = <K extends keyof Presentation>(key: K, value: Presentation[K]) =>
    setPresentation((prev) => ({ ...prev, [key]: value }))

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(presentation))
    } catch {
      // A full or blocked storage quota must not break sharing.
    }
  }, [presentation])

  const models = useMemo(() => cardModels(modelRows), [modelRows])
  const stats = useMemo(
    () =>
      availableStats({ totals, series, hideDollars: presentation.hideDollars }),
    [totals, series, presentation.hideDollars],
  )
  const hero = resolveHero(stats, presentation.hero)
  // Memoized because both feed the rasterize effect's dependency list. As fresh
  // arrays on every render they made that effect re-run after its own setPreview,
  // which re-armed the 300ms debounce and re-encoded the card on a loop for as
  // long as the dialog stayed open.
  const secondary = useMemo(
    () => stats.filter((stat) => stat.id !== hero?.id).slice(0, 3),
    [stats, hero?.id],
  )
  const shown = useMemo(
    () => models.slice(0, presentation.rows),
    [models, presentation.rows],
  )
  const scope = `${windowLabel} · UTC${scopeSuffix}`

  // Rasterize the card the user will actually post, debounced so dragging a
  // control or typing a title does not re-encode a 1080px PNG per keystroke.
  //
  // hero/secondary/shown/scope are not read here: they are what the card renders,
  // so the effect depends on them through the DOM it rasterizes. Dropping them
  // would leave the PNG showing the previous card.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the dependency is the rendered card, which the rule cannot see
  useEffect(() => {
    let cancelled = false
    let url: string | undefined
    const timer = setTimeout(() => {
      const node = cardRef.current
      if (node === null) {
        return
      }
      const { width, height } = CARD_SIZES[presentation.ratio]
      rasterize(node, { width, height })
        .then((blob) => {
          if (cancelled) {
            return
          }
          url = URL.createObjectURL(blob)
          setPreview((previous) => {
            if (previous !== undefined) {
              URL.revokeObjectURL(previous)
            }
            return url
          })
          setError(undefined)
        })
        .catch((cause: unknown) => {
          if (!cancelled) {
            setError(
              cause instanceof Error
                ? cause
                : new Error("The share card could not be rendered."),
            )
          }
        })
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [presentation, hero, secondary, shown, scope])

  // The last preview URL outlives the effect above, so it is revoked on unmount.
  useEffect(
    () => () => {
      setPreview((previous) => {
        if (previous !== undefined) {
          URL.revokeObjectURL(previous)
        }
        return undefined
      })
    },
    [],
  )

  const copyable = canCopyImages()
  const blocked = isStale || hero === null

  async function withBlob(
    action: (blob: Blob) => Promise<void> | void,
    label: string,
  ) {
    const node = cardRef.current
    if (node === null) {
      return
    }
    setBusy(true)
    setError(undefined)
    try {
      const { width, height } = CARD_SIZES[presentation.ratio]
      const blob = await rasterize(node, { width, height })
      await action(blob)
      setNotice(label)
      clearTimeout(noticeTimer.current)
      noticeTimer.current = setTimeout(() => setNotice(undefined), 2000)
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause : new Error("That did not work."))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal isOpen onOpenChange={(open) => (open ? undefined : onClose())}>
      <Modal.Trigger aria-hidden="true" className="hidden">
        Open share dialog
      </Modal.Trigger>
      <Modal.Backdrop>
        <Modal.Container placement="center" size="lg" scroll="outside">
          <Modal.Dialog className="otari-modal-wide">
            <Modal.Header>
              <Modal.Heading>Share this view as an image</Modal.Heading>
            </Modal.Header>
            <Modal.Body className="flex flex-col gap-4">
              <p className="text-caption">
                The card shows the window and filters currently applied above.
                Change them on the page to change what it says.
              </p>

              <ErrorBanner error={error} />

              {isStale ? (
                <InfoBanner tone="warning">
                  Waiting for the current numbers before this can be shared.
                </InfoBanner>
              ) : null}

              <div className="flex flex-col gap-4 sm:flex-row">
                {/* The preview is the PNG itself at feed width, not a styled DOM stand-in:
            what is approved here is byte-for-byte what gets posted. */}
                <div className="w-full sm:w-[340px] sm:shrink-0">
                  {preview !== undefined ? (
                    <img
                      src={preview}
                      alt="Preview of the usage card that will be shared"
                      className="h-auto w-full rounded-md border border-border"
                    />
                  ) : (
                    <div className="flex aspect-square w-full items-center justify-center rounded-md border border-dashed border-border text-caption">
                      Rendering preview…
                    </div>
                  )}
                </div>

                <div className="flex min-w-0 flex-1 flex-col gap-4">
                  <Field label="Lead with">
                    <div className="inline-flex flex-wrap gap-1.5">
                      {heroCandidates(stats).map((stat) => (
                        <Button
                          key={stat.id}
                          size="sm"
                          variant={hero?.id === stat.id ? "primary" : "outline"}
                          aria-pressed={hero?.id === stat.id}
                          onPress={() => set("hero", stat.id)}
                        >
                          {stat.label}
                        </Button>
                      ))}
                    </div>
                  </Field>

                  <Field
                    label={`Title (${presentation.title.length}/${TITLE_MAX})`}
                  >
                    <input
                      value={presentation.title}
                      maxLength={TITLE_MAX}
                      onChange={(event) => set("title", event.target.value)}
                      className="w-full rounded-md border border-border px-2 py-1 text-sm"
                      aria-label="Card title"
                    />
                  </Field>

                  <div className="flex flex-wrap gap-4">
                    <Field label="Shape">
                      <div className="inline-flex gap-1.5">
                        {(["square", "landscape"] as CardRatio[]).map(
                          (ratio) => (
                            <Button
                              key={ratio}
                              size="sm"
                              variant={
                                presentation.ratio === ratio
                                  ? "primary"
                                  : "outline"
                              }
                              aria-pressed={presentation.ratio === ratio}
                              onPress={() => set("ratio", ratio)}
                            >
                              {ratio === "square" ? "Square" : "Wide"}
                            </Button>
                          ),
                        )}
                      </div>
                    </Field>
                    <Field label="Theme">
                      <div className="inline-flex gap-1.5">
                        {(["dark", "light"] as const).map((theme) => (
                          <Button
                            key={theme}
                            size="sm"
                            variant={
                              presentation.theme === theme
                                ? "primary"
                                : "outline"
                            }
                            aria-pressed={presentation.theme === theme}
                            onPress={() => set("theme", theme)}
                          >
                            {theme === "dark" ? "Dark" : "Light"}
                          </Button>
                        ))}
                      </div>
                    </Field>
                    <Field label="Model rows">
                      <div className="inline-flex gap-1.5">
                        {ROW_CHOICES.map((rows) => (
                          <Button
                            key={rows}
                            size="sm"
                            variant={
                              presentation.rows === rows ? "primary" : "outline"
                            }
                            aria-pressed={presentation.rows === rows}
                            onPress={() => set("rows", rows)}
                          >
                            {rows}
                          </Button>
                        ))}
                      </div>
                    </Field>
                  </div>

                  <div className="flex flex-wrap gap-1.5">
                    <Button
                      size="sm"
                      variant={presentation.hideDollars ? "primary" : "outline"}
                      aria-pressed={presentation.hideDollars}
                      onPress={() =>
                        set("hideDollars", !presentation.hideDollars)
                      }
                    >
                      Hide dollar amounts
                    </Button>
                  </div>
                </div>
              </div>
            </Modal.Body>
            <Modal.Footer className="flex flex-wrap items-center gap-2">
              {notice !== undefined ? (
                <span className="mr-auto text-xs text-accent">{notice}</span>
              ) : null}
              <Button variant="ghost" onPress={onClose}>
                Close
              </Button>
              <Button
                variant={copyable ? "outline" : "primary"}
                isDisabled={busy || blocked}
                onPress={() =>
                  withBlob(
                    (blob) =>
                      downloadBlob(blob, shareFilename(startIso, endIso)),
                    "Image saved",
                  )
                }
              >
                Download PNG
              </Button>
              {copyable ? (
                <Button
                  variant="primary"
                  isDisabled={busy || blocked}
                  onPress={() =>
                    withBlob(async (blob) => {
                      if (!(await copyBlobAsImage(blob))) {
                        throw new Error(
                          "The image could not be copied to the clipboard.",
                        )
                      }
                    }, "Image copied")
                  }
                >
                  Copy image
                </Button>
              ) : null}
            </Modal.Footer>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>

      {/* Rendered at full size off-screen: html cannot be rasterized from a
          display:none subtree (it has no layout), and the preview above must not
          be the scaled node itself or the PNG would inherit the scale. */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed -left-[9999px] top-0 opacity-0"
      >
        <div ref={cardRef}>
          <ShareCard
            ratio={presentation.ratio}
            theme={presentation.theme}
            title={presentation.title}
            scope={scope}
            hero={hero}
            models={shown}
            stats={secondary}
            unpricedRequests={totals?.unpriced_requests}
          />
        </div>
      </div>
    </Modal>
  )
}

function Field({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-caption">{label}</span>
      {children}
    </div>
  )
}
