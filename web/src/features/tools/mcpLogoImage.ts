import type { McpLogoTone } from "@/features/tools/McpServerLogo"

export type McpLogoTile = "auto" | "always" | "never"

// The gateway's own limits and thresholds (services/mcp_catalog/_logos.py), so
// the preview tiles a logo exactly when the saved one will be.
export const LOGO_MAX_BYTES = 512 * 1024
const LOGO_SIZE = 256
const LOGO_TYPES = new Set(["image/png", "image/svg+xml"])
const DARK_LOGO_BELOW = 0.35
const LIGHT_LOGO_ABOVE = 0.85

export interface PreparedLogo {
  png: Blob
  previewUrl: string
  /** How bright the ink is, 0 to 1; undefined when it cannot be told. */
  luminance: number | undefined
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () =>
      reject(new Error("That file is not an image this browser can read."))
    image.src = url
  })
}

/**
 * Rec. 709 luma weighed by opacity, as the gateway measures a logo. A logo on a
 * square of its own (an opaque edge) needs no tile, so it has none.
 */
export function inkLuminance(
  pixels: Uint8ClampedArray,
  size: number,
): number | undefined {
  let edge = 0
  let opaqueEdge = 0
  let total = 0
  let weight = 0
  for (let index = 0; index < pixels.length; index += 4) {
    const pixel = index / 4
    const x = pixel % size
    const y = Math.floor(pixel / size)
    const alpha = pixels[index + 3]
    if (x === 0 || y === 0 || x === size - 1 || y === size - 1) {
      edge += 1
      if (alpha >= 250) opaqueEdge += 1
    }
    const luma =
      0.2126 * pixels[index] +
      0.7152 * pixels[index + 1] +
      0.0722 * pixels[index + 2]
    total += (luma / 255) * alpha
    weight += alpha
  }
  if (edge > 0 && opaqueEdge / edge >= 0.75) return undefined
  return weight ? total / weight : undefined
}

/**
 * A PNG or SVG an admin picked, center-cropped to a square and drawn at 256 px,
 * ready to upload as a server's logo. The gateway checks it again.
 */
export async function prepareLogo(file: File): Promise<PreparedLogo> {
  if (!LOGO_TYPES.has(file.type)) throw new Error("Pick a PNG or SVG file.")
  if (file.size > LOGO_MAX_BYTES) throw new Error("Pick a file under 512 KB.")
  const sourceUrl = URL.createObjectURL(file)
  try {
    const image = await loadImage(sourceUrl)
    // An SVG without a width or height reports zero; draw it at full size.
    const width = image.naturalWidth || LOGO_SIZE
    const height = image.naturalHeight || LOGO_SIZE
    const side = Math.min(width, height)
    const canvas = document.createElement("canvas")
    canvas.width = LOGO_SIZE
    canvas.height = LOGO_SIZE
    const context = canvas.getContext("2d")
    if (!context) throw new Error("This browser cannot resize images.")
    context.imageSmoothingQuality = "high"
    context.drawImage(
      image,
      (width - side) / 2,
      (height - side) / 2,
      side,
      side,
      0,
      0,
      LOGO_SIZE,
      LOGO_SIZE,
    )
    const luminance = inkLuminance(
      context.getImageData(0, 0, LOGO_SIZE, LOGO_SIZE).data,
      LOGO_SIZE,
    )
    const png = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("Could not make the PNG.")),
        "image/png",
      ),
    )
    if (png.size > LOGO_MAX_BYTES) throw new Error("Pick a simpler image.")
    return { png, previewUrl: URL.createObjectURL(png), luminance }
  } finally {
    URL.revokeObjectURL(sourceUrl)
  }
}

/** The tile a logo gets, as the gateway decides it. */
export function logoTone(
  tile: McpLogoTile,
  luminance: number | undefined,
): McpLogoTone | undefined {
  if (tile === "never") return undefined
  if (tile === "always") return "tile"
  if (luminance === undefined) return undefined
  if (luminance < DARK_LOGO_BELOW) return "dark"
  if (luminance > LIGHT_LOGO_ABOVE) return "light"
  return undefined
}
