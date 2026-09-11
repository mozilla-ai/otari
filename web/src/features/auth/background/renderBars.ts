import { barLuminance } from "./barWave"
import { BAR_ROW_RATIO, type LoginBackgroundConfig } from "./config"

export interface BarGeometry {
  width: number
  height: number
  left: number
  top: number
  panelHeight: number
  panelWidth: number
}

export interface BarPalette {
  background: string
  accent: string
}

export function drawBars(
  ctx: CanvasRenderingContext2D,
  geometry: BarGeometry,
  palette: BarPalette,
  config: LoginBackgroundConfig,
  time: number,
) {
  const { width, height, left, top, panelWidth, panelHeight } = geometry
  if (panelWidth <= 0 || width <= 0 || height <= 0) return
  const pitchX = panelWidth / config.columns
  const pitchY = pitchX * BAR_ROW_RATIO
  const barWidth = pitchX * (1 - config.spacing)
  const barHeight = pitchY * (1 - config.spacing * (0.134 / 0.248))
  const offsetX = (pitchX - barWidth) / 2
  const offsetY = (pitchY - barHeight) / 2
  const radius = barWidth * config.rounding

  ctx.globalAlpha = 1
  ctx.fillStyle = palette.background
  ctx.fillRect(0, 0, width, height)
  ctx.fillStyle = palette.accent

  for (
    let row = Math.floor(-top / pitchY);
    row < Math.ceil((height - top) / pitchY);
    row++
  ) {
    for (
      let col = Math.floor(-left / pitchX);
      col < Math.ceil((width - left) / pitchX);
      col++
    ) {
      const x = left + col * pitchX
      const y = top + row * pitchY
      // The opaque form hides complete cells inside its bounds.
      if (
        x >= left &&
        x + pitchX <= left + panelWidth &&
        y >= top &&
        y + pitchY <= top + panelHeight
      )
        continue
      const luminance = barLuminance(
        (x + pitchX / 2) / width,
        (y + pitchY / 2) / height,
        time,
        config.waveScale,
      )
      ctx.globalAlpha = luminance ** config.contrast * config.intensity
      ctx.beginPath()
      ctx.roundRect(x + offsetX, y + offsetY, barWidth, barHeight, radius)
      ctx.fill()
    }
  }
  ctx.globalAlpha = 1
}
