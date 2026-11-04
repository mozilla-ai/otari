import { formatNumber, formatTokens } from "@/shared/helpers/format"
import type { CardModel, CardStat } from "./shareCardData"

// The card is rasterized to a PNG and posted publicly, which makes it a different
// medium from the dashboard it draws on, and the differences are load-bearing:
//
//  - Feed scale, not full size. A single image renders around 500px wide in a
//    timeline and as little as 130px in a grid or a quoted post. Nothing here may
//    fall below FLOOR_PX at 1080, which is roughly 13px once the feed scales it
//    down. That rule is what caps the payload at a hero, three model rows, and
//    three secondary stats: everything else was unreadable anyway.
//  - Its own palette, not the app's tokens. `--color-surface` is #ffffff, so a
//    token-driven card dissolves into a light-mode timeline and reads as exactly
//    the screenshot this feature exists to beat. These constants are card-local on
//    purpose: the foundation's dark theme dresses the dashboard on a dark display,
//    while these two dress an image posted into someone else's timeline, so the
//    card's `theme` prop is the poster's choice and not the viewer's.
//  - Literal colors, no `var(--color-*)`. Custom properties do not resolve inside
//    an SVG document loaded through an `<img>` element, which is how this is
//    rasterized, so a token reference here renders as nothing. This file is the
//    one documented exception to the no-raw-hex rule in foundation.test.ts.

export const CARD_SIZES = {
  square: { width: 1080, height: 1080 },
  landscape: { width: 1200, height: 630 },
} as const

export type CardRatio = keyof typeof CARD_SIZES

/** Minimum type size at 1080 scale. Below this, the feed's downscale makes it noise. */
const FLOOR_PX = 28

/**
 * Approximate the width of `text`, in em, at the card's font.
 *
 * The card is rendered off-screen and rasterized in the same tick, so there is
 * nothing to measure: a layout pass would have to happen between deciding the
 * type size and drawing it. These per-class averages are tuned on the formatted
 * numbers the hero sets, where they track Inter to within ~1%, but the card is
 * rasterized in whatever the viewer's stack resolves and no font is bundled, so
 * the error against a real frame runs to ~10% low on prose and ~19% high on digits
 * for the common fallbacks. Callers own that margin: size *down* from this and the
 * error is slack, size a clipping box from it and the error costs content.
 */
export function emWidth(text: string): number {
  return [...text].reduce((em, char) => em + charEmWidth(char), 0)
}

function charEmWidth(char: string): number {
  if (
    char === "," ||
    char === "." ||
    char === " " ||
    char === "'" ||
    char === "|"
  ) {
    return 0.28
  }
  if (char === "-" || char === "*") return 0.35
  if (char === "$") return 0.58
  if (char >= "0" && char <= "9") return 0.6
  if (char >= "A" && char <= "Z") return 0.65
  if (char >= "a" && char <= "z") return 0.52
  return 0.6
}

/**
 * Twice the legibility floor: the point past which a value has stopped reading as
 * the headline. It covers every value this card can actually be handed, including
 * eleven digits of request count in the narrower wide column. A value long enough
 * to be clamped *up* by this floor does overflow, so it is set as low as legibility
 * allows: reaching it takes roughly $100T of spend, which no formatter here emits.
 */
const MIN_HERO = FLOOR_PX * 2

/**
 * The largest type size at which `text` still fits `available` px, capped at `cap`.
 *
 * The hero is user data with no length bound: a hobby gateway's "$4.20" and a
 * team's "1,204,881" go in the same slot. A single fixed size cannot serve both,
 * and the one that was here (200px) put "$2,390.99" 23px past the edge of a square
 * card and clean off a wide one. Size follows the value instead, so the cap is
 * what a *short* value gets and a long one steps down from there.
 */
export function fitHeroSize(
  text: string,
  available: number,
  cap: number,
): number {
  if (text.length === 0) {
    return cap
  }
  // 1.02 covers the estimator's error; MIN_HERO keeps a freak value legible rather
  // than letting it shrink until it stops being the headline.
  const fitted = Math.floor(available / (emWidth(text) * 1.02))
  return Math.max(MIN_HERO, Math.min(cap, fitted))
}

/** Both palettes share this shape; `as const` alone would make them incompatible types. */
export interface CardPalette {
  ground: string
  panel: string
  ink: string
  muted: string
  bar: string
  track: string
  rule: string
}

const DARK: CardPalette = {
  ground: "#0d1a20",
  panel: "#14242c",
  ink: "#f2f7f8",
  muted: "#8fa8b2",
  bar: "#5fb0c9",
  track: "#22343d",
  rule: "#22343d",
}

const LIGHT: CardPalette = {
  ground: "#f6f9fa",
  panel: "#ffffff",
  ink: "#14242c",
  muted: "#5b7280",
  bar: "#4e8295",
  track: "#e6edef",
  rule: "#dbe5e8",
}

export interface ShareCardProps {
  ratio: CardRatio
  theme: "dark" | "light"
  title: string
  /** Names the window and any active filters, so the denominator is never ambiguous. */
  scope: string
  hero: CardStat | undefined
  /** Up to three, already token-ranked. */
  models: CardModel[]
  /** Up to three, excluding whichever one is the hero. */
  stats: CardStat[]
  /**
   * Requests in the window with no configured price. Printed as the asterisk's
   * legend when a caveated stat is shown: the card is posted as a standalone
   * file, so a mark whose meaning lives only in the docs is unreadable.
   */
  unpricedRequests?: number
}

/**
 * Middle-truncate a model id.
 *
 * Model ids are distinguished by their tails (`claude-sonnet-4-5-20260514`), so
 * right-truncation destroys the only part that identifies them.
 */
export function truncateModel(name: string, max = 28): string {
  if (name.length <= max) {
    return name
  }
  const head = Math.ceil((max - 1) / 2)
  const tail = Math.floor((max - 1) / 2)
  return `${name.slice(0, head)}…${name.slice(name.length - tail)}`
}

// The Otari mark, inlined from web/public/favicon.svg rather than referenced.
// The card is rasterized by loading it as an <img>, and that document cannot
// fetch anything external, so an <img src="/favicon.svg"> would silently render
// nothing. The fill is passed in because the accent disappears on the dark card
// ground, which is also why this one cannot use `ProductMark`: that takes its
// color from the page, and this document has no page to take it from.
//
// So this is a hand-kept copy of the path, and the thing that catches it going
// stale is `ShareCard.test.tsx` asserting the viewBox, which is why that
// assertion names the number rather than just checking a mark is present.
function OtariMark({ color, height }: { color: string; height: number }) {
  return (
    <svg
      width={(height * 273) / 250}
      height={height}
      viewBox="0 0 273 250"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M107.045 214.396C106.677 213.589 106.314 212.785 105.959 211.989C103.686 206.885 101.689 202.029 99.937 197.474C90.768 173.642 88.318 158.027 88.296 157.885L105.738 187.422L106.633 188.922L121.56 214.778C121.854 215.288 122.162 215.805 122.468 216.305C122.665 216.627 122.866 216.949 123.068 217.268C125.307 220.799 127.82 224.061 130.565 227.041C135.129 231.996 140.331 236.174 145.966 239.517C166.651 251.788 193.182 252.828 215.493 239.947L215.245 239.517L188.478 193.153C184.608 186.452 179.78 180.647 174.268 175.813C175.385 175.929 186.021 177.967 214.771 213.759L214.803 213.802L214.958 213.993L214.945 213.663C246.233 196.444 268.099 164.217 270.829 126.677C271.194 124.473 271.383 122.209 271.383 119.901L271.383 26.599C248.645 26.599 230.212 45.03 230.212 67.769C230.212 45.03 211.788 26.599 189.05 26.599L189.05 77.714C189.05 88.784 193.42 98.835 200.528 106.234L201.807 107.505C208.33 113.728 216.892 117.833 226.39 118.71L146.435 118.71L116.488 66.743C145.509 59.437 166.993 33.169 166.994 1.881L166.994 0L77.785 0.02C51.671 -0.019 30.449 21.142 30.448 47.288L30.448 134.572C30.449 142.999 31.577 151.163 33.69 158.921C34.889 163.324 36.406 167.597 38.213 171.713C47.518 192.902 64.541 209.935 85.725 219.252C89.758 221.026 93.942 222.52 98.252 223.71C103.437 225.141 108.805 226.132 114.313 226.641C113.063 224.831 111.877 222.954 110.758 221.016L107.424 215.241C107.365 215.108 107.303 214.973 107.241 214.835C107.214 214.774 107.187 214.712 107.159 214.65C107.122 214.567 107.084 214.482 107.045 214.396ZM29.396 182.229L28.673 183.507L28.679 183.529C28.111 184.418 27.551 185.319 27.019 186.241L0 233.036C24.653 247.269 54.458 244.503 75.836 228.374C75.698 228.304 75.557 228.234 75.419 228.164L75.894 227.837C75.86 227.819 52.19 215.667 42.286 191.177C39.261 183.696 37.566 176.917 36.765 173.188L36.319 170.966C36.226 170.462 36.184 170.185 36.182 170.173L29.396 182.229ZM100.172 12.059C105.917 12.059 110.572 16.717 110.572 22.462C110.572 28.206 105.917 32.862 100.172 32.863C94.427 32.863 89.768 28.207 89.768 22.462C89.768 16.716 94.427 12.059 100.172 12.059Z"
        fill={color}
      />
    </svg>
  )
}

export function ShareCard(props: ShareCardProps) {
  const { ratio, theme, title, scope, hero, models, stats, unpricedRequests } =
    props
  const palette = theme === "dark" ? DARK : LIGHT
  const { width, height } = CARD_SIZES[ratio]
  const isLandscape = ratio === "landscape"
  const maxTokens = models.reduce(
    (most, model) => Math.max(most, model.tokens),
    0,
  )

  // The two shapes are not one layout at two sizes. A wide card has 550px of usable
  // height against a square card's 936, and the square's arrangement (hero, then
  // rows, then a full-width stats band) needs about 800 of them, so the wide card
  // ran its hero over the title above and its rows through the rule below. Wide
  // splits into columns instead: the claim (hero plus its supporting stats) on the
  // left, the evidence (the model rows) on the right, which gives the rows the
  // whole middle height and gives the numbers a column narrow enough to size type
  // against. Every constant below is one of the two shapes' real dimensions, and
  // the row height is what is left after the fixed bands are subtracted, so a
  // change to any of them moves the rows rather than silently overflowing.
  const pad = isLandscape ? 40 : 72
  const contentWidth = width - pad * 2
  const contentHeight = height - pad * 2
  const bandGap = isLandscape ? 24 : 32
  const titleSize = isLandscape ? 36 : 40
  // Reserved rather than measured: the title is free text (up to 60 characters),
  // and a slot that grew with it would push the hero down by a line. Two lines is
  // the cap in both shapes; a wide card fits 60 characters on one.
  //
  // The wrap is decided at 0.85 of the real width because the slot clips what it
  // does not reserve, and the estimator runs low on prose: its per-class averages
  // are numeric-accurate but undercount lowercase-heavy text by ~10% against the
  // Arial-metric stacks this falls back to, which on a wide card silently dropped
  // the second line of a 60-character title. Reserving a line that goes unused
  // costs the rows some height; clipping one loses the words.
  const titleLines = Math.min(
    2,
    Math.max(
      1,
      Math.ceil((emWidth(title) * titleSize) / (contentWidth * 0.85)),
    ),
  )
  // An exact pixel line height, not a unitless 1.2, so the slot is a whole number
  // of line boxes rather than of a value the browser resolved for itself. The 4px
  // is for ink, not layout: a font whose descenders run past their own line box
  // (Liberation Sans does, by 3px at this size) would otherwise have the tail of a
  // second-line "g" clipped by the slot that hides a third line.
  const titleLine = Math.round(titleSize * 1.2)
  const titleSlot = titleLines * titleLine + 4
  const footerSlot = 40
  /** The "tokens per model" caption plus the gap above it. */
  const captionSlot = 42

  const rowCount = Math.max(models.length, 1)
  const heroLabelSize = isLandscape ? 36 : 44
  // The wide card's hero column, sized so the widest supporting stat line
  // ("$2,391  AVG LATENCY") still fits it.
  const heroColumn = 400
  // A long list needs the hero to give up some height; nothing else on a square
  // card can yield.
  const heroCap = isLandscape ? 120 : rowCount > 5 ? 132 : 168
  const heroSize = fitHeroSize(
    hero?.value ?? "",
    isLandscape ? heroColumn : contentWidth,
    heroCap,
  )
  const heroBlock =
    Math.round(heroSize * 0.95) + 10 + Math.round(heroLabelSize * 1.1)

  // Wide keeps its stats beside the hero, so only the square card spends a band on
  // them. Both counts are subtracted here so the rows get exactly what is left.
  const hasStatsBand = !isLandscape && stats.length > 0
  const statsBandHeight = hasStatsBand
    ? 2 + 32 + Math.round(64 * 1.05) + Math.round(FLOOR_PX * 1.2)
    : 0
  const bandCount = hasStatsBand ? 4 : 3
  const middleHeight =
    contentHeight -
    titleSlot -
    footerSlot -
    statsBandHeight -
    bandGap * (bandCount - 1)
  // On a wide card the hero sits beside the rows and costs them no height at all.
  const rowsArea =
    middleHeight -
    captionSlot -
    (isLandscape || hero === undefined ? 0 : heroBlock + 32)

  const rowGap = rowCount > 5 ? 8 : rowCount > 3 ? 12 : 18
  // Floored at 34 so the 28px name still has room, capped so a single row does not
  // become a band. The floor deliberately wins over the budget: at the tightest
  // combination the surplus comes out of the band gaps rather than shrinking a row
  // below its own text. The e2e asserts no band collapses, which is what keeps that
  // trade honest if these numbers ever move.
  const rowHeight = Math.max(
    34,
    Math.min(
      isLandscape ? 44 : 56,
      Math.floor((rowsArea - rowGap * (rowCount - 1)) / rowCount),
    ),
  )
  // The rows' fixed columns. Narrower on a wide card, where the row column is 680px
  // rather than the square's 936 and the bar is what would otherwise vanish: it did,
  // taking the token counts off the edge of the card with it.
  const nameWidth = isLandscape ? 330 : 360
  const nameMax = isLandscape ? 22 : 28
  const valueWidth = isLandscape ? 100 : 120
  const rowInnerGap = isLandscape ? 16 : 20

  // Only when a caveated stat is actually on the card, so the legend never
  // explains a mark the viewer cannot see.
  const showsCaveat =
    (hero?.isCaveated ?? false) || stats.some((stat) => stat.isCaveated)
  const caveatLegend: string | undefined =
    showsCaveat && unpricedRequests !== undefined && unpricedRequests > 0
      ? `* ${formatNumber(unpricedRequests)} requests unpriced`
      : showsCaveat
        ? "* some requests unpriced"
        : undefined

  // On a wide card this column carries the whole claim: the hero and, under it,
  // the stats a square card gives a band of its own.
  const claimColumn = (
    <div
      style={{
        // The fixed column is the hero's, so the empty state is not made to wrap
        // inside a width chosen for a number that is not there.
        flex:
          isLandscape && hero !== undefined
            ? `0 0 ${heroColumn}px`
            : "0 0 auto",
        width: isLandscape && hero !== undefined ? heroColumn : undefined,
      }}
    >
      {hero !== undefined ? (
        <>
          <div
            style={{ fontSize: heroSize, fontWeight: 700, lineHeight: 0.95 }}
          >
            {hero.value}
          </div>
          <div
            style={{
              fontSize: heroLabelSize,
              fontWeight: 500,
              lineHeight: 1.1,
              color: palette.muted,
              marginTop: 10,
            }}
          >
            {hero.label}
            {hero.isCaveated ? "*" : ""}
          </div>
        </>
      ) : (
        <div style={{ fontSize: 44, color: palette.muted }}>
          No usage in this range
        </div>
      )}

      {isLandscape && stats.length > 0 ? (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            marginTop: 28,
          }}
        >
          {stats.map((stat) => (
            <div
              key={stat.id}
              style={{ display: "flex", alignItems: "baseline", gap: 12 }}
            >
              <span style={{ fontSize: 40, fontWeight: 600, lineHeight: 1.05 }}>
                {stat.value}
                {stat.isCaveated ? "*" : ""}
              </span>
              <span
                style={{
                  fontSize: FLOOR_PX,
                  color: palette.muted,
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  whiteSpace: "nowrap",
                }}
              >
                {stat.label}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )

  const rowsColumn =
    models.length === 0 ? null : (
      <div
        style={{
          flex: isLandscape ? "1 1 0" : "0 0 auto",
          minWidth: 0,
          width: isLandscape ? undefined : "100%",
          display: "flex",
          flexDirection: "column",
          gap: rowGap,
        }}
      >
        {models.map((model) => (
          <div
            key={model.key ?? "__other__"}
            data-share-row
            style={{
              display: "flex",
              alignItems: "center",
              gap: rowInnerGap,
              height: rowHeight,
            }}
          >
            <span
              style={{
                fontSize: FLOOR_PX,
                width: nameWidth,
                flexShrink: 0,
                whiteSpace: "nowrap",
              }}
            >
              {truncateModel(model.label, nameMax)}
            </span>
            <span
              style={{
                flex: "1 1 0",
                minWidth: 0,
                height: Math.min(26, Math.round(rowHeight * 0.46)),
                background: palette.track,
                borderRadius: 13,
              }}
            >
              <span
                style={{
                  display: "block",
                  height: "100%",
                  borderRadius: 10,
                  width:
                    maxTokens > 0
                      ? `${(model.tokens / maxTokens) * 100}%`
                      : "0%",
                  background: palette.bar,
                }}
              />
            </span>
            <span
              style={{
                fontSize: FLOOR_PX,
                color: palette.muted,
                width: valueWidth,
                flexShrink: 0,
                textAlign: "right",
              }}
            >
              {formatTokens(model.tokens)}
            </span>
          </div>
        ))}
        <div style={{ fontSize: FLOOR_PX, color: palette.muted }}>
          tokens per model
        </div>
      </div>
    )

  return (
    <div
      role="img"
      aria-label={`Usage card: ${title}`}
      style={{
        width,
        height,
        padding: pad,
        boxSizing: "border-box",
        background: palette.ground,
        color: palette.ink,
        display: "flex",
        flexDirection: "column",
        gap: bandGap,
        fontFamily: "'Inter', system-ui, sans-serif",
      }}
    >
      {/* Fixed-height title slot so a one- or two-line title never shifts the
          hero below it. */}
      <div
        style={{
          flexShrink: 0,
          height: titleSlot,
          fontSize: titleSize,
          fontWeight: 600,
          lineHeight: `${titleLine}px`,
          overflow: "hidden",
        }}
      >
        {title}
      </div>

      {/* The hero and the model rows are one unit: the claim and its evidence.
          This block takes all the card's leftover height (flex: 1) and centres
          that unit inside it, so the slack collects above and below rather than
          being shared out between the number and the rows that explain it, which
          is what `justify-content: space-between` on the card would otherwise do.
          The wide card's two columns live in one inner row, so they stay top
          aligned with each other while the pair still centres as a unit: centring
          each column on its own left a single model row floating in the middle of
          an otherwise empty half. */}
      <div
        style={{
          flex: "1 1 auto",
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: isLandscape ? "row" : "column",
            alignItems: "flex-start",
            gap: isLandscape ? 40 : 32,
          }}
        >
          {claimColumn}
          {rowsColumn}
        </div>
      </div>

      {hasStatsBand ? (
        <div
          style={{
            flexShrink: 0,
            display: "flex",
            gap: 64,
            borderTop: `2px solid ${palette.rule}`,
            paddingTop: 32,
          }}
        >
          {stats.map((stat) => (
            <div key={stat.id}>
              <div style={{ fontSize: 64, fontWeight: 600, lineHeight: 1.05 }}>
                {stat.value}
                {stat.isCaveated ? "*" : ""}
              </div>
              <div
                style={{
                  fontSize: FLOOR_PX,
                  color: palette.muted,
                  textTransform: "uppercase",
                  letterSpacing: 1,
                }}
              >
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      <div
        style={{
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: FLOOR_PX,
          color: palette.muted,
        }}
      >
        <span>
          {scope}
          {caveatLegend !== undefined ? ` · ${caveatLegend}` : ""}
        </span>
        {/* Hardcoded, never derived from `location`: a self-hosted gateway would
            otherwise publish its own internal hostname onto a public image. */}
        <span
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            color: palette.ink,
            fontWeight: 600,
          }}
        >
          otari.ai
          <OtariMark color={palette.bar} height={36} />
        </span>
      </div>
    </div>
  )
}
