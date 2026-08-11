import type { CardModel, CardStat } from "@/lib/shareCard";
import { formatNumber, formatTokens } from "@/lib/format";

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
  hero: CardStat | null;
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
  // The card is a fixed frame, so the row list has a fixed height budget and the
  // rows divide it. Hard-coded row heights broke both ways: at three rows a square
  // card left ~350px of blank space that read as a hole, and at nine rows the
  // content overflowed and flex-shrink collapsed the title to zero height.
  // Dividing a budget instead means every row count in the picker (1/3/5/9)
  // renders, in both shapes, with the surplus staying ordinary margin.
  const rowCount = Math.max(models.length, 1);
  const rowsBudget = isLandscape ? 210 : 340;
  const rowGap = rowCount > 5 ? 8 : rowCount > 3 ? 12 : 18;
  // Floored at 34 so the 28px name still has room, capped so a single row does not
  // become a band.
  const rowHeight = Math.max(34, Math.min(isLandscape ? 40 : 56, Math.floor((rowsBudget - rowGap * (rowCount - 1)) / rowCount)));
  // A long list needs the hero to give up some height; nothing else can yield.
  const heroSize = rowCount > 5 ? 150 : 200;
  // Only when a caveated stat is actually on the card, so the legend never
  // explains a mark the viewer cannot see.
  const showsCaveat = (hero?.caveated ?? false) || stats.some((stat) => stat.caveated);
  const caveatLegend =
    showsCaveat && unpricedRequests !== undefined && unpricedRequests > 0
      ? `* ${formatNumber(unpricedRequests)} requests unpriced`
      : showsCaveat
        ? "* some requests unpriced"
        : null;

  return (
    <div
      data-testid="share-card"
      style={{
        width,
        height,
        padding: 72,
        boxSizing: "border-box",
        background: palette.ground,
        color: palette.ink,
        display: "flex",
        flexDirection: "column",
        gap: 40,
        fontFamily: "'Inter', system-ui, sans-serif",
      }}
    >
      {/* Fixed-height title slot so a one- or two-line title never shifts the
          hero below it. */}
      <div style={{ flexShrink: 0, maxHeight: 96, fontSize: 40, fontWeight: 600, lineHeight: 1.2, overflow: "hidden" }}>{title}</div>

      {/* The hero and the model rows are one unit: the claim and its evidence.
          This block takes all the card's leftover height (flex: 1) and centres
          that unit inside it, so the slack collects above and below rather than
          being shared out between the number and the rows that explain it, which
          is what `justify-content: space-between` on the card would otherwise do. */}
      <div
        style={{
          flex: "1 1 auto",
          minHeight: 0,
          display: "flex",
          flexDirection: isLandscape ? "row" : "column",
          justifyContent: "center",
          gap: isLandscape ? 48 : 32,
          alignItems: isLandscape ? "center" : "flex-start",
        }}
      >
        {hero !== null ? (
          <div style={{ flex: isLandscape ? "1 1 0" : "0 0 auto" }}>
            <div style={{ fontSize: heroSize, fontWeight: 700, lineHeight: 0.82 }}>{hero.value}</div>
            <div style={{ fontSize: 44, fontWeight: 500, lineHeight: 1.1, color: palette.muted, marginTop: 8 }}>
              {hero.label}
              {hero.caveated ? "*" : ""}
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 44, color: palette.muted }}>No usage in this range</div>
        )}

        <div style={{ flex: isLandscape ? "1 1 0" : "0 0 auto", width: "100%", display: "flex", flexDirection: "column", gap: 32 }}>
          {models.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: rowGap, width: "100%" }}>
              {models.map((model) => (
                <div key={model.key ?? "other"} style={{ display: "flex", alignItems: "center", gap: 20, height: rowHeight }}>
                  <span style={{ fontSize: FLOOR_PX, width: 360, whiteSpace: "nowrap" }}>
                    {truncateModel(model.label)}
                  </span>
                  <span style={{ flex: 1, height: Math.min(26, Math.round(rowHeight * 0.46)), background: palette.track, borderRadius: 13 }}>
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
                  <span style={{ fontSize: FLOOR_PX, color: palette.muted, width: 120, textAlign: "right" }}>
                    {formatTokens(model.tokens)}
                  </span>
                </div>
              ))}
              <div style={{ fontSize: FLOOR_PX, color: palette.muted }}>tokens per model</div>
            </div>
          ) : null}
        </div>
      </div>

      {stats.length > 0 ? (
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
          {caveatLegend !== null ? ` · ${caveatLegend}` : ""}
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
