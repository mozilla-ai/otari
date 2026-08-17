// Rasterize a DOM node to a PNG with no third-party dependency.
//
// The route is: serialize the node -> wrap it in an SVG <foreignObject> -> load
// that as an image -> draw it to a canvas -> canvas.toBlob("image/png"). Three
// details are load-bearing and each one is a bug if done the obvious way:
//
//  1. Serialize with XMLSerializer, never string templates. An unescaped "&" in a
//     user's title ("R&D usage") or a model name makes the SVG unparseable, and
//     the only symptom is img.onerror with no diagnostic.
//  2. Encode with a Blob, never btoa. btoa throws InvalidCharacterError on any
//     non-Latin-1 character, and the title field is free text, so one emoji would
//     break rasterization.
//  3. The node must carry every style inline and use literal colors. Custom
//     properties (var(--otari-*)) do not resolve inside an <img>-loaded SVG
//     document, so a token reference renders as nothing. ShareCard is built to
//     this constraint deliberately.

const LOAD_TIMEOUT_MS = 15_000

export interface RasterizeOptions {
  width: number
  height: number
  /** Device-pixel multiplier. 2 gives a crisp card on a retina timeline. */
  pixelRatio?: number
}

/** True when this context can put an image on the clipboard at all. */
export function canCopyImages(): boolean {
  return (
    typeof window !== "undefined" &&
    window.isSecureContext &&
    typeof ClipboardItem !== "undefined" &&
    typeof navigator !== "undefined" &&
    navigator.clipboard !== undefined &&
    "write" in navigator.clipboard
  )
}

function serializeToSvg(
  node: HTMLElement,
  width: number,
  height: number,
): string {
  // Clone so the live node keeps its own attributes, and give the clone the XHTML
  // namespace a foreignObject requires.
  const clone = node.cloneNode(true) as HTMLElement
  clone.setAttribute("xmlns", "http://www.w3.org/1999/xhtml")

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
  svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
  svg.setAttribute("width", String(width))
  svg.setAttribute("height", String(height))
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`)

  const foreign = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "foreignObject",
  )
  foreign.setAttribute("width", "100%")
  foreign.setAttribute("height", "100%")
  foreign.appendChild(clone)
  svg.appendChild(foreign)

  return new XMLSerializer().serializeToString(svg)
}

/** Rasterize `node` to a PNG blob at exactly `width` x `height` logical pixels. */
export async function rasterize(
  node: HTMLElement,
  options: RasterizeOptions,
): Promise<Blob> {
  const { width, height, pixelRatio = 2 } = options
  const svg = serializeToSvg(node, width, height)
  // A data: URI, NOT a blob: URL. Chromium taints a canvas that has drawn an SVG
  // image loaded from a blob URL, and a tainted canvas refuses toBlob() outright
  // ("Tainted canvases may not be exported"), so the blob-URL route can never
  // produce a PNG. encodeURIComponent rather than btoa keeps non-Latin-1 titles
  // (an emoji) working.
  const svgUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`

  const image = new Image()
  image.width = width
  image.height = height
  await new Promise<void>((resolve, reject) => {
    // Bounded: the promise otherwise settles only on load or error, and a decode
    // that fires neither would leave the dialog busy with both actions disabled
    // and nothing said. A timeout turns that into the normal error path.
    const timer = setTimeout(
      () => reject(new Error("Rendering the share card timed out.")),
      LOAD_TIMEOUT_MS,
    )
    image.onload = () => {
      clearTimeout(timer)
      resolve()
    }
    image.onerror = () => {
      clearTimeout(timer)
      reject(new Error("The share card could not be rendered to an image."))
    }
    image.src = svgUrl
  })

  const canvas = document.createElement("canvas")
  canvas.width = Math.round(width * pixelRatio)
  canvas.height = Math.round(height * pixelRatio)
  const ctx = canvas.getContext("2d")
  if (ctx === null) {
    throw new Error(
      "This browser did not provide a 2D canvas to render the card.",
    )
  }
  ctx.scale(pixelRatio, pixelRatio)
  ctx.drawImage(image, 0, 0, width, height)

  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob === null) {
        reject(new Error("The rendered card could not be encoded as a PNG."))
      } else {
        resolve(blob)
      }
    }, "image/png")
  })
}

/**
 * Filename for a downloaded card.
 *
 * Carries the whole window rather than a single date: a single date mislabels
 * every multi-day range, which is most of them.
 */
export function shareFilename(startIso: string, endIso: string): string {
  // The window can be absent before the first summary resolves, and an unguarded
  // slice produced "otari-usage-.png". The date is dropped rather than
  // substituted: stamping today onto a card covering some other window would
  // mislabel the file, which is worse than not labelling it.
  const day = (iso: string) => (iso.length >= 10 ? iso.slice(0, 10) : undefined)
  const from = day(startIso)
  const to = day(endIso)
  if (from === undefined || to === undefined) {
    return "otari-usage.png"
  }
  return from === to
    ? `otari-usage-${from}.png`
    : `otari-usage-${from}--${to}.png`
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.rel = "noopener"
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // Revoked well after the click, not synchronously: an immediate revoke can race
  // the download in some browsers, which then saves a zero-byte file.
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

export async function copyBlobAsImage(blob: Blob): Promise<boolean> {
  if (!canCopyImages()) {
    return false
  }
  try {
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })])
    return true
  } catch {
    return false
  }
}
