import { AlertDialog, Button } from "@heroui/react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { UsageGroupRow, UsageSeriesPoint, UsageTotals } from "@/api/types";
import { availableStats, cardModels, heroCandidates, resolveHero, type StatId } from "@/lib/shareCard";
import { canCopyImages, copyBlobAsImage, downloadBlob, rasterize, shareFilename } from "@/lib/shareImage";

import { CARD_SIZES, ShareCard, type CardRatio } from "./ShareCard";

// Presentation choices persist so a returning user does not re-pick them. The data
// scope deliberately does NOT: the page's current filters are the user's current
// question, and restoring a stale window from storage would silently change what
// the card claims. That is why there is no window or filter control in here.
const STORE_KEY = "otari.share.presentation.v1";

interface Presentation {
  hero: StatId;
  ratio: CardRatio;
  theme: "dark" | "light";
  hideDollars: boolean;
  rows: number;
  title: string;
}

const DEFAULTS: Presentation = {
  hero: "requests",
  ratio: "square",
  theme: "dark",
  hideDollars: false,
  rows: 5,
  title: "Where my tokens went",
};

function loadPresentation(): Presentation {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw === null) {
      return DEFAULTS;
    }
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) {
      return DEFAULTS;
    }
    // Merged over the defaults rather than trusted: a stored shape from an older
    // build (or a hand-edited one) must not be able to crash the panel.
    return { ...DEFAULTS, ...(parsed as Partial<Presentation>) };
  } catch {
    return DEFAULTS;
  }
}

const TITLE_MAX = 60;

export interface ShareDialogProps {
  totals: UsageTotals | undefined;
  series: UsageSeriesPoint[];
  /** `by_model` rows for the page's current filters, straight off the page's own query. */
  modelRows: UsageGroupRow[];
  windowLabel: string;
  /** Names any active entity filters, so the card's denominator is unambiguous. */
  scopeSuffix: string;
  startIso: string;
  endIso: string;
  /** True while the page's own query is in flight or showing placeholder data. */
  isStale: boolean;
  onClose: () => void;
}

export function ShareDialog(props: ShareDialogProps) {
  const { totals, series, modelRows, windowLabel, scopeSuffix, startIso, endIso, isStale, onClose } = props;
  const [presentation, setPresentation] = useState<Presentation>(loadPresentation);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  const set = <K extends keyof Presentation>(key: K, value: Presentation[K]) =>
    setPresentation((prev) => ({ ...prev, [key]: value }));

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(presentation));
    } catch {
      // A full or blocked storage quota must not break sharing.
    }
  }, [presentation]);

  const models = useMemo(() => cardModels(modelRows), [modelRows]);
  const stats = useMemo(
    () => availableStats({ totals, series, hideDollars: presentation.hideDollars }),
    [totals, series, presentation.hideDollars],
  );
  const hero = resolveHero(stats, presentation.hero);
  const secondary = stats.filter((stat) => stat.id !== hero?.id).slice(0, 3);
  const shown = models.slice(0, presentation.rows);
  const scope = `${windowLabel} · UTC${scopeSuffix}`;

  // Rasterize the card the user will actually post, debounced so dragging a
  // control or typing a title does not re-encode a 1080px PNG per keystroke.
  useEffect(() => {
    let cancelled = false;
    let url: string | null = null;
    const timer = setTimeout(() => {
      const node = cardRef.current;
      if (node === null) {
        return;
      }
      const { width, height } = CARD_SIZES[presentation.ratio];
      rasterize(node, { width, height })
        .then((blob) => {
          if (cancelled) {
            return;
          }
          url = URL.createObjectURL(blob);
          setPreview((previous) => {
            if (previous !== null) {
              URL.revokeObjectURL(previous);
            }
            return url;
          });
          setError(null);
        })
        .catch((cause: unknown) => {
          if (!cancelled) {
            setError(cause instanceof Error ? cause.message : "The share card could not be rendered.");
          }
        });
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [presentation, hero, secondary, shown, scope]);

  // The last preview URL outlives the effect above, so it is revoked on unmount.
  useEffect(
    () => () => {
      setPreview((previous) => {
        if (previous !== null) {
          URL.revokeObjectURL(previous);
        }
        return null;
      });
    },
    [],
  );

  const copyable = canCopyImages();
  const blocked = isStale || hero === null;

  async function withBlob(action: (blob: Blob) => Promise<void> | void, label: string) {
    const node = cardRef.current;
    if (node === null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { width, height } = CARD_SIZES[presentation.ratio];
      const blob = await rasterize(node, { width, height });
      await action(blob);
      setNotice(label);
      setTimeout(() => setNotice(null), 2000);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "That did not work.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AlertDialog isOpen onOpenChange={(open) => (open ? undefined : onClose())}>
      <AlertDialog.Backdrop>
        <AlertDialog.Container placement="center" size="lg">
          <AlertDialog.Dialog className="w-[92vw] max-w-[940px]">
            <AlertDialog.Header>
              <AlertDialog.Heading>Share this view as an image</AlertDialog.Heading>
            </AlertDialog.Header>
            <AlertDialog.Body className="flex max-h-[70vh] flex-col gap-4 overflow-y-auto">
      <p className="text-xs text-[var(--otari-muted)]">
        The card shows the window and filters currently applied above. Change them on the page to change what it says.
      </p>

      {error !== null ? (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">{error}</div>
      ) : null}

      {isStale ? (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Waiting for the current numbers before this can be shared.
        </div>
      ) : null}

      <div className="flex flex-col gap-4 sm:flex-row">
        {/* The preview is the PNG itself at feed width, not a styled DOM stand-in:
            what is approved here is byte-for-byte what gets posted. */}
        <div className="w-full sm:w-[340px] sm:shrink-0">
          {preview !== null ? (
            <img
              src={preview}
              alt="Preview of the usage card that will be shared"
              className="h-auto w-full rounded-md border border-[var(--otari-line)]"
            />
          ) : (
            <div className="flex aspect-square w-full items-center justify-center rounded-md border border-dashed border-[var(--otari-line)] text-xs text-[var(--otari-muted)]">
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
                  onPress={() => set("hero", stat.id)}
                >
                  {stat.label}
                </Button>
              ))}
            </div>
          </Field>

          <Field label={`Title (${presentation.title.length}/${TITLE_MAX})`}>
            <input
              value={presentation.title}
              maxLength={TITLE_MAX}
              onChange={(event) => set("title", event.target.value)}
              className="w-full rounded-md border border-[var(--otari-line)] px-2 py-1 text-sm"
              aria-label="Card title"
            />
          </Field>

          <div className="flex flex-wrap gap-4">
            <Field label="Shape">
              <div className="inline-flex gap-1.5">
                {(["square", "landscape"] as CardRatio[]).map((ratio) => (
                  <Button
                    key={ratio}
                    size="sm"
                    variant={presentation.ratio === ratio ? "primary" : "outline"}
                    onPress={() => set("ratio", ratio)}
                  >
                    {ratio === "square" ? "Square" : "Wide"}
                  </Button>
                ))}
              </div>
            </Field>
            <Field label="Theme">
              <div className="inline-flex gap-1.5">
                {(["dark", "light"] as const).map((theme) => (
                  <Button
                    key={theme}
                    size="sm"
                    variant={presentation.theme === theme ? "primary" : "outline"}
                    onPress={() => set("theme", theme)}
                  >
                    {theme === "dark" ? "Dark" : "Light"}
                  </Button>
                ))}
              </div>
            </Field>
            <Field label="Model rows">
              <div className="inline-flex gap-1.5">
                {[1, 3, 5, 9].map((rows) => (
                  <Button
                    key={rows}
                    size="sm"
                    variant={presentation.rows === rows ? "primary" : "outline"}
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
              onPress={() => set("hideDollars", !presentation.hideDollars)}
            >
              Hide dollar amounts
            </Button>
          </div>

        </div>
      </div>

            </AlertDialog.Body>
            <AlertDialog.Footer className="flex flex-wrap items-center gap-2">
              {notice !== null ? <span className="mr-auto text-xs text-[var(--otari-brand)]">{notice}</span> : null}
              <Button variant="ghost" onPress={onClose}>
                Close
              </Button>
              <Button
                variant={copyable ? "outline" : "primary"}
                isDisabled={busy || blocked}
                onPress={() => withBlob((blob) => downloadBlob(blob, shareFilename(startIso, endIso)), "Image saved")}
              >
                Download PNG
              </Button>
              {copyable ? (
                <Button
                  variant="primary"
                  isDisabled={busy || blocked}
                  onPress={() => withBlob(async (blob) => { await copyBlobAsImage(blob); }, "Image copied")}
                >
                  Copy image
                </Button>
              ) : null}
            </AlertDialog.Footer>
          </AlertDialog.Dialog>
        </AlertDialog.Container>
      </AlertDialog.Backdrop>

      {/* Rendered at full size off-screen: html cannot be rasterized from a
          display:none subtree (it has no layout), and the preview above must not
          be the scaled node itself or the PNG would inherit the scale. */}
      <div aria-hidden className="pointer-events-none fixed -left-[9999px] top-0 opacity-0">
        <div ref={cardRef}>
          <ShareCard
            ratio={presentation.ratio}
            theme={presentation.theme}
            title={presentation.title}
            scope={scope}
            hero={hero}
            models={shown}
            stats={secondary}
          />
        </div>
      </div>
    </AlertDialog>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-[var(--otari-muted)]">{label}</span>
      {children}
    </div>
  );
}
