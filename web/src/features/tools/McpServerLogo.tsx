import { useState } from "react"

import { useResolvedTheme } from "@/shared/hooks/useTheme"

// A server's logo, which the gateway fetched once and serves from its own
// origin, or an admin uploaded; else the server's first letter.
//
// Drawn bare, like the dashboard's own icons. A tile goes behind it only on the
// theme where it would vanish, as the gateway measured when it stored it: dark
// ink gets a light tile on the dark theme, light ink a dark tile on the light
// theme. An admin can ask for a tile everywhere, or never. The tile borrows the
// other theme's own surface through `data-theme`, so it needs no color of its
// own.

export type McpLogoTone = "dark" | "light" | "tile"
export type McpLogoTheme = "light" | "dark"

const SIZE = {
  sm: "size-7 text-caption",
  md: "size-9 text-sm",
  lg: "size-11 text-title",
} as const

/** Whether a logo of this tone sits on a tile on this theme, and which theme's surface the tile takes. */
export function logoTile(
  tone: McpLogoTone | null | undefined,
  theme: McpLogoTheme,
): { tiled: false } | { tiled: true; surface: McpLogoTheme | undefined } {
  if (tone === "tile") return { tiled: true, surface: undefined }
  if (tone === "dark" && theme === "dark") {
    return { tiled: true, surface: "light" }
  }
  if (tone === "light" && theme === "light") {
    return { tiled: true, surface: "dark" }
  }
  return { tiled: false }
}

export function McpServerLogo({
  name,
  logoUrl,
  tone,
  size = "md",
  theme,
}: {
  /** What the letter is taken from when there is no image. */
  name: string
  /** The gateway's URL for the logo; null or "" draws the letter. */
  logoUrl: string | null | undefined
  tone?: McpLogoTone | null
  size?: keyof typeof SIZE
  /** The theme to draw for, where it is not the page's: a preview of the other one. */
  theme?: McpLogoTheme
}) {
  const pageTheme = useResolvedTheme()
  // The URL that failed, so a new one gets its own chance to load.
  const [failed, setFailed] = useState("")
  const url = logoUrl ?? ""
  const showImage = url !== "" && failed !== url
  const letter = name.trim().charAt(0).toUpperCase() || "?"
  if (!showImage) {
    return (
      <span
        // Decorative: the name is on screen beside it everywhere this renders.
        aria-hidden
        className={`flex shrink-0 items-center justify-center overflow-hidden rounded-md border border-control-border bg-surface-alt text-muted ${SIZE[size]}`}
      >
        {letter}
      </span>
    )
  }
  const tile = logoTile(tone, theme ?? pageTheme)
  return (
    <span
      aria-hidden
      data-theme={tile.tiled ? tile.surface : undefined}
      data-logo-tile={tile.tiled ? (tone ?? undefined) : undefined}
      className={`flex shrink-0 items-center justify-center overflow-hidden rounded-md ${SIZE[size]} ${
        tile.tiled ? "border border-border bg-surface-alt p-1" : ""
      }`}
    >
      <img
        src={url}
        alt=""
        loading="lazy"
        decoding="async"
        draggable={false}
        className="size-full object-contain"
        onError={() => setFailed(url)}
      />
    </span>
  )
}
