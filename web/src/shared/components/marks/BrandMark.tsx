import { ProductMark } from "@/design-system/ProductMark"
import {
  hasProviderMark,
  isProductMarkProvider,
  makerMark,
  type ProviderMarkGlyph,
  providerMark,
} from "@/shared/helpers/brandMarks"
import { providerDisplayName } from "@/shared/helpers/providers"

/**
 * The two steps a mark is drawn at.
 *
 * 16 sits beside the 14px body roles, 14 beside the 13px mono rows and the
 * caption lines. Both are declared here rather than taken as a number, so a
 * third size is a decision somebody makes in the design rather than a call site
 * inventing one.
 */
export type BrandMarkStep = 16 | 14

const BOX: Record<BrandMarkStep, string> = {
  16: "size-4",
  14: "size-3.5",
}

/**
 * The geometry, drawn.
 *
 * Decorative, so it is hidden from the accessibility tree rather than labelled:
 * a mark is never shown without the name it belongs to, and a label here would
 * read the company out twice.
 */
function Glyph({ glyph, box }: { glyph: ProviderMarkGlyph; box: string }) {
  const shapes = glyph.shapes.map((shape) => (
    <path key={shape.d} d={shape.d} fillOpacity={shape.fillOpacity} />
  ))
  return (
    <svg
      viewBox={glyph.viewBox}
      aria-hidden="true"
      focusable="false"
      className={`${box} shrink-0 fill-current`}
    >
      {glyph.transform === undefined ? (
        shapes
      ) : (
        <g transform={glyph.transform}>{shapes}</g>
      )}
    </svg>
  )
}

/**
 * The stand-in for a company with no mark: its initial in a bordered box.
 *
 * A tile rather than a blank, because a missing mark is the ordinary case and
 * not a fault. An operator names their own instances, and some ids have no mark
 * anyone would recognize.
 */
function Lettermark({ label, box }: { label: string; box: string }) {
  return (
    <span
      aria-hidden="true"
      // `bg-surface-alt` is the registered utility for `--color-surface-muted`;
      // `bg-surface-muted` compiles to nothing (see `Login`'s CODE_CHIP).
      // `text-shell-monogram` carries the 9px step and nothing else, so the
      // family and the weight are set here, as `Avatar` sets its own. That role
      // is borrowed rather than declared again at a third name: its docblock's
      // reasoning is this tile's exactly, one glyph in a small box being
      // recognized rather than read. If the "shell" in it reads wrong here,
      // renaming the role is the fix.
      //
      // Close to `design-system/indicators/Avatar`, deliberately not it: that
      // one is 20px or 26px with a sans semibold monogram, where these are 16px
      // and 14px in mono, and reaching for it would mean overriding a
      // primitive's own size from a call site.
      className={`${box} text-shell-monogram flex shrink-0 items-center justify-center border border-control-border bg-surface-alt font-mono font-medium uppercase`}
    >
      {label.trim().slice(0, 1)}
    </span>
  )
}

/**
 * A provider's own mark, or a lettermark tile when it has none.
 *
 * Always drawn beside the provider's name, never instead of it, which is what
 * lets sibling ids share a mark: `azureopenai` and `azureanthropic` are both
 * Azure credentials and carry the Azure mark, and the name is what tells them
 * apart.
 *
 * Monochrome and inheriting the page's text ink. See `brandMarks` for why a
 * per-vendor brand color is not an option.
 *
 * Whether a surface shows tiles at all is the caller's call, not this
 * component's, and the rule is per list: if no row on a surface resolves a
 * mark, no row gets a slot; if any row does, every row carries one and the
 * unresolved rows tile.
 */
export function ProviderMark({
  providerId,
  step = 16,
  label,
}: {
  /** The provider id or instance name the mark is keyed on. */
  providerId: string
  step?: BrandMarkStep
  /**
   * The label shown beside the mark, whose first character the tile takes. The
   * displayed label rather than the id, so a renamed instance tiles as the
   * operator spelled it.
   */
  label?: string
}) {
  const box = BOX[step]

  // Our own models take the product mark. It is not square (273x250), and the
  // default `preserveAspectRatio` fits it inside the box rather than stretching
  // it, so it shares the step without sharing the aspect.
  if (isProductMarkProvider(providerId)) {
    return <ProductMark className={`${box} shrink-0`} />
  }

  const glyph = providerMark(providerId)
  if (glyph === undefined) {
    return (
      <Lettermark label={label ?? providerDisplayName(providerId)} box={box} />
    )
  }
  return <Glyph glyph={glyph} box={box} />
}

/**
 * A model maker's mark, or a lettermark tile when it has none.
 *
 * Keyed on the vendor slug rather than the vendor's display string, because the
 * catalog id already carries that slug and recomputing it here would mirror the
 * gateway's normalization in a second place. `makerKeyOf` in the catalog module
 * is what reads it off a row.
 */
export function MakerMark({
  vendorSlug,
  label,
  step = 14,
}: {
  /** The vendor slug from the catalog id, e.g. `mistralai` or `z-ai`. */
  vendorSlug: string
  /** The maker as it reads on the row, whose first character the tile takes. */
  label: string
  step?: BrandMarkStep
}) {
  const box = BOX[step]
  const glyph = makerMark(vendorSlug)
  return glyph === undefined ? (
    <Lettermark label={label} box={box} />
  ) : (
    <Glyph glyph={glyph} box={box} />
  )
}

/**
 * Whether any of these providers has a mark, which is what decides if a list
 * reserves the slot at all.
 *
 * A list where nothing resolves would otherwise be a column of identical
 * tiles, which carries no information and costs every row 16px of indent.
 */
export function anyProviderMark(providerIds: readonly string[]): boolean {
  return providerIds.some(hasProviderMark)
}

/** The same question for a list of makers. */
export function anyMakerMark(vendorSlugs: readonly string[]): boolean {
  return vendorSlugs.some((slug) => makerMark(slug) !== undefined)
}
