import type { CardModel, CardStat } from "./shareCardData";
import { formatNumber, formatTokens } from "@/shared/lib/format";

// The card is rasterized to a PNG and posted publicly, which makes it a different
// medium from the dashboard it draws on, and the differences are load-bearing:
//
//  - Feed scale, not full size. A single image renders around 500px wide in a
//    timeline and as little as 130px in a grid or a quoted post. Nothing here may
//    fall below FLOOR_PX at 1080, which is roughly 13px once the feed scales it
//    down. That rule is what caps the payload at a hero, three model rows, and
//    three secondary stats: everything else was unreadable anyway.
//  - Its own palette, not the app's tokens. `--otari-surface` is #ffffff, so a
//    token-driven card dissolves into a light-mode timeline and reads as exactly
//    the screenshot this feature exists to beat. These constants are card-local on
//    purpose: they are not a dashboard dark mode (the app has no dark token set)
//    and must not grow into one.
//  - Literal colors, no `var(--otari-*)`. Custom properties do not resolve inside
//    an SVG document loaded through an `<img>` element, which is how this is
//    rasterized, so a token reference here renders as nothing.

export const CARD_SIZES = {
  square: { width: 1080, height: 1080 },
  landscape: { width: 1200, height: 630 },
} as const;

export type CardRatio = keyof typeof CARD_SIZES;

/** Minimum type size at 1080 scale. Below this, the feed's downscale makes it noise. */
const FLOOR_PX = 28;

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
  let em = 0;
  for (const char of text) {
    if (char === "," || char === "." || char === " " || char === "'" || char === "|") {
      em += 0.28;
    } else if (char === "-" || char === "*") {
      em += 0.35;
    } else if (char === "$") {
      em += 0.58;
    } else if (char >= "0" && char <= "9") {
      em += 0.6;
    } else if (char >= "A" && char <= "Z") {
      em += 0.65;
    } else if (char >= "a" && char <= "z") {
      em += 0.52;
    } else {
      em += 0.6;
    }
  }
  return em;
}

/**
 * Twice the legibility floor: the point past which a value has stopped reading as
 * the headline. It covers every value this card can actually be handed, including
 * eleven digits of request count in the narrower wide column. A value long enough
 * to be clamped *up* by this floor does overflow, so it is set as low as legibility
 * allows: reaching it takes roughly $100T of spend, which no formatter here emits.
 */
const MIN_HERO = FLOOR_PX * 2;

/**
 * The largest type size at which `text` still fits `available` px, capped at `cap`.
 *
 * The hero is user data with no length bound: a hobby gateway's "$4.20" and a
 * team's "1,204,881" go in the same slot. A single fixed size cannot serve both,
 * and the one that was here (200px) put "$2,390.99" 23px past the edge of a square
 * card and clean off a wide one. Size follows the value instead, so the cap is
 * what a *short* value gets and a long one steps down from there.
 */
export function fitHeroSize(text: string, available: number, cap: number): number {
  if (text.length === 0) {
    return cap;
  }
  // 1.02 covers the estimator's error; MIN_HERO keeps a freak value legible rather
  // than letting it shrink until it stops being the headline.
  const fitted = Math.floor(available / (emWidth(text) * 1.02));
  return Math.max(MIN_HERO, Math.min(cap, fitted));
}

/** Both palettes share this shape; `as const` alone would make them incompatible types. */
export interface CardPalette {
  ground: string;
  panel: string;
  ink: string;
  muted: string;
  bar: string;
  track: string;
  rule: string;
}

const DARK: CardPalette = {
  ground: "#0d1a20",
  panel: "#14242c",
  ink: "#f2f7f8",
  muted: "#8fa8b2",
  bar: "#5fb0c9",
  track: "#22343d",
  rule: "#22343d",
};

const LIGHT: CardPalette = {
  ground: "#f6f9fa",
  panel: "#ffffff",
  ink: "#14242c",
  muted: "#5b7280",
  bar: "#4e8295",
  track: "#e6edef",
  rule: "#dbe5e8",
};

export interface ShareCardProps {
  ratio: CardRatio;
  theme: "dark" | "light";
  title: string;
  /** Names the window and any active filters, so the denominator is never ambiguous. */
  scope: string;
  hero: CardStat | undefined;
  /** Up to three, already token-ranked. */
  models: CardModel[];
  /** Up to three, excluding whichever one is the hero. */
  stats: CardStat[];
  /**
   * Requests in the window with no configured price. Printed as the asterisk's
   * legend when a caveated stat is shown: the card is posted as a standalone
   * file, so a mark whose meaning lives only in the docs is unreadable.
   */
  unpricedRequests?: number;
}

/**
 * Middle-truncate a model id.
 *
 * Model ids are distinguished by their tails (`claude-sonnet-4-5-20260514`), so
 * right-truncation destroys the only part that identifies them.
 */
export function truncateModel(name: string, max = 28): string {
  if (name.length <= max) {
    return name;
  }
  const head = Math.ceil((max - 1) / 2);
  const tail = Math.floor((max - 1) / 2);
  return `${name.slice(0, head)}…${name.slice(name.length - tail)}`;
}

// The Otari mark, inlined from web/public/favicon.svg rather than referenced.
// The card is rasterized by loading it as an <img>, and that document cannot
// fetch anything external, so an <img src="/favicon.svg"> would silently render
// nothing. The fill is passed in because the brand blue disappears on the dark
// card ground.
function OtariMark({ color, height }: { color: string; height: number }) {
  return (
    <svg
      width={(height * 272) / 250}
      height={height}
      viewBox="0 0 272 250"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path d="M193.598 0.0195312C219.712 -0.0192587 240.934 21.1417 240.935 47.2881V134.572C240.934 142.999 239.806 151.163 237.693 158.921C236.494 163.324 234.977 167.597 233.17 171.713C223.865 192.902 206.842 209.935 185.658 219.252C181.625 221.026 177.441 222.52 173.131 223.71C167.946 225.141 162.578 226.132 157.07 226.641C158.32 224.831 159.506 222.954 160.625 221.016L163.959 215.241C164.018 215.108 164.08 214.973 164.142 214.835C164.169 214.774 164.196 214.712 164.224 214.65C164.261 214.567 164.299 214.482 164.338 214.396C164.706 213.589 165.069 212.785 165.424 211.989C167.697 206.885 169.694 202.029 171.446 197.474C180.615 173.642 183.065 158.027 183.087 157.885L165.645 187.422L164.75 188.922L149.823 214.778C149.529 215.288 149.221 215.805 148.915 216.305C148.718 216.627 148.517 216.949 148.315 217.268C146.076 220.799 143.563 224.061 140.818 227.041C136.254 231.996 131.052 236.174 125.417 239.517C104.732 251.788 78.201 252.828 55.8896 239.947L56.1377 239.517L82.9053 193.153C86.7745 186.452 91.6032 180.647 97.1152 175.813C95.9976 175.929 85.3616 177.967 56.6123 213.759L56.5801 213.802L56.4248 213.993L56.4385 213.663C25.1498 196.444 3.2838 164.217 0.553711 126.677C0.188854 124.473 3.79607e-05 122.209 0 119.901V26.5986C22.7381 26.5986 41.1708 45.0304 41.1709 67.7686C41.171 45.0304 59.5949 26.5986 82.333 26.5986V77.7139C82.3328 88.7841 77.9626 98.8355 70.8545 106.234L69.5762 107.505C63.0535 113.728 54.4913 117.833 44.9932 118.71H124.948L154.895 66.7432C125.874 59.4372 104.39 33.1686 104.389 1.88086V0L193.598 0.0195312ZM241.987 182.229L242.71 183.507L242.704 183.529C243.272 184.418 243.832 185.319 244.364 186.241L271.383 233.036C246.73 247.269 216.925 244.503 195.547 228.374C195.685 228.304 195.826 228.234 195.964 228.164L195.489 227.837C195.523 227.819 219.193 215.667 229.097 191.177C232.122 183.696 233.817 176.917 234.618 173.188L235.064 170.966C235.157 170.462 235.199 170.185 235.201 170.173L241.987 182.229ZM171.211 12.0586C165.466 12.0595 160.811 16.7171 160.811 22.4619C160.811 28.2064 165.466 32.8624 171.211 32.8633C176.956 32.8633 181.615 28.207 181.615 22.4619C181.615 16.7165 176.956 12.0586 171.211 12.0586Z" fill={color} />
    </svg>
  );
}

export function ShareCard(props: ShareCardProps) {
  const { ratio, theme, title, scope, hero, models, stats, unpricedRequests } = props;
  const palette = theme === "dark" ? DARK : LIGHT;
  const { width, height } = CARD_SIZES[ratio];
  const isLandscape = ratio === "landscape";
  const maxTokens = models.reduce((most, model) => Math.max(most, model.tokens), 0);

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
  const pad = isLandscape ? 40 : 72;
  const contentWidth = width - pad * 2;
  const contentHeight = height - pad * 2;
  const bandGap = isLandscape ? 24 : 32;
  const titleSize = isLandscape ? 36 : 40;
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
  const titleLines = Math.min(2, Math.max(1, Math.ceil((emWidth(title) * titleSize) / (contentWidth * 0.85))));
  // An exact pixel line height, not a unitless 1.2, so the slot is a whole number
  // of line boxes rather than of a value the browser resolved for itself. The 4px
  // is for ink, not layout: a font whose descenders run past their own line box
  // (Liberation Sans does, by 3px at this size) would otherwise have the tail of a
  // second-line "g" clipped by the slot that hides a third line.
  const titleLine = Math.round(titleSize * 1.2);
  const titleSlot = titleLines * titleLine + 4;
  const footerSlot = 40;
  /** The "tokens per model" caption plus the gap above it. */
  const captionSlot = 42;

  const rowCount = Math.max(models.length, 1);
  const heroLabelSize = isLandscape ? 36 : 44;
  // The wide card's hero column, sized so the widest supporting stat line
  // ("$2,391  AVG LATENCY") still fits it.
  const heroColumn = 400;
  // A long list needs the hero to give up some height; nothing else on a square
  // card can yield.
  const heroCap = isLandscape ? 120 : rowCount > 5 ? 132 : 168;
  const heroSize = fitHeroSize(hero?.value ?? "", isLandscape ? heroColumn : contentWidth, heroCap);
  const heroBlock = Math.round(heroSize * 0.95) + 10 + Math.round(heroLabelSize * 1.1);

  // Wide keeps its stats beside the hero, so only the square card spends a band on
  // them. Both counts are subtracted here so the rows get exactly what is left.
  const hasStatsBand = !isLandscape && stats.length > 0;
  const statsBandHeight = hasStatsBand ? 2 + 32 + Math.round(64 * 1.05) + Math.round(FLOOR_PX * 1.2) : 0;
  const bandCount = hasStatsBand ? 4 : 3;
  const middleHeight = contentHeight - titleSlot - footerSlot - statsBandHeight - bandGap * (bandCount - 1);
  // On a wide card the hero sits beside the rows and costs them no height at all.
  const rowsArea = middleHeight - captionSlot - (isLandscape || hero === undefined ? 0 : heroBlock + 32);

  const rowGap = rowCount > 5 ? 8 : rowCount > 3 ? 12 : 18;
  // Floored at 34 so the 28px name still has room, capped so a single row does not
  // become a band. The floor deliberately wins over the budget: at the tightest
  // combination the surplus comes out of the band gaps rather than shrinking a row
  // below its own text. The e2e asserts no band collapses, which is what keeps that
  // trade honest if these numbers ever move.
  const rowHeight = Math.max(
    34,
    Math.min(isLandscape ? 44 : 56, Math.floor((rowsArea - rowGap * (rowCount - 1)) / rowCount)),
  );
  // The rows' fixed columns. Narrower on a wide card, where the row column is 680px
  // rather than the square's 936 and the bar is what would otherwise vanish: it did,
  // taking the token counts off the edge of the card with it.
  const nameWidth = isLandscape ? 330 : 360;
  const nameMax = isLandscape ? 22 : 28;
  const valueWidth = isLandscape ? 100 : 120;
  const rowInnerGap = isLandscape ? 16 : 20;

  // Only when a caveated stat is actually on the card, so the legend never
  // explains a mark the viewer cannot see.
  const showsCaveat = (hero?.caveated ?? false) || stats.some((stat) => stat.caveated);
  const caveatLegend: string | undefined =
    showsCaveat && unpricedRequests !== undefined && unpricedRequests > 0
      ? `* ${formatNumber(unpricedRequests)} requests unpriced`
      : showsCaveat
        ? "* some requests unpriced"
        : undefined;

  // On a wide card this column carries the whole claim: the hero and, under it,
  // the stats a square card gives a band of its own.
  const claimColumn = (
    <div
      style={{
        // The fixed column is the hero's, so the empty state is not made to wrap
        // inside a width chosen for a number that is not there.
        flex: isLandscape && hero !== undefined ? `0 0 ${heroColumn}px` : "0 0 auto",
        width: isLandscape && hero !== undefined ? heroColumn : undefined,
      }}
    >
      {hero !== undefined ? (
        <>
          <div style={{ fontSize: heroSize, fontWeight: 700, lineHeight: 0.95 }}>{hero.value}</div>
          <div
            style={{ fontSize: heroLabelSize, fontWeight: 500, lineHeight: 1.1, color: palette.muted, marginTop: 10 }}
          >
            {hero.label}
            {hero.caveated ? "*" : ""}
          </div>
        </>
      ) : (
        <div style={{ fontSize: 44, color: palette.muted }}>No usage in this range</div>
      )}

      {isLandscape && stats.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 28 }}>
          {stats.map((stat) => (
            <div key={stat.id} style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
              <span style={{ fontSize: 40, fontWeight: 600, lineHeight: 1.05 }}>
                {stat.value}
                {stat.caveated ? "*" : ""}
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
  );

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
            style={{ display: "flex", alignItems: "center", gap: rowInnerGap, height: rowHeight }}
          >
            <span style={{ fontSize: FLOOR_PX, width: nameWidth, flexShrink: 0, whiteSpace: "nowrap" }}>
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
                  width: maxTokens > 0 ? `${(model.tokens / maxTokens) * 100}%` : "0%",
                  background: palette.bar,
                }}
              />
            </span>
            <span
              style={{ fontSize: FLOOR_PX, color: palette.muted, width: valueWidth, flexShrink: 0, textAlign: "right" }}
            >
              {formatTokens(model.tokens)}
            </span>
          </div>
        ))}
        <div style={{ fontSize: FLOOR_PX, color: palette.muted }}>tokens per model</div>
      </div>
    );

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
        <div style={{ flexShrink: 0, display: "flex", gap: 64, borderTop: `2px solid ${palette.rule}`, paddingTop: 32 }}>
          {stats.map((stat) => (
            <div key={stat.id}>
              <div style={{ fontSize: 64, fontWeight: 600, lineHeight: 1.05 }}>
                {stat.value}
                {stat.caveated ? "*" : ""}
              </div>
              <div style={{ fontSize: FLOOR_PX, color: palette.muted, textTransform: "uppercase", letterSpacing: 1 }}>
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
        <span style={{ display: "flex", alignItems: "center", gap: 14, color: palette.ink, fontWeight: 600 }}>
          <OtariMark color={palette.bar} height={36} />
          otari.ai
        </span>
      </div>
    </div>
  );
}
