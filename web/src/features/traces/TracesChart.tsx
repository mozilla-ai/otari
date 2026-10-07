import type { TraceSeries } from "@/client"
import { formatNumber } from "@/design-system/helpers/format"
import {
  type SeriesDef,
  type StackedPoint,
  TrendChart,
} from "@/design-system/metrics/charts"

const SERIES: SeriesDef[] = [
  { key: "succeeded", label: "Succeeded", color: "var(--color-primary)" },
  { key: "failed", label: "Failed", color: "var(--color-danger)" },
]

function tick(iso: string, bucket: TraceSeries["bucket"]): string {
  const date = new Date(iso)
  return bucket === "hour"
    ? date.toLocaleTimeString(undefined, { hour: "numeric" })
    : date.toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

// Sessions started per bucket, split by whether any of their spans failed with
// nothing recovering it.
export function TracesChart({ series }: { series: TraceSeries }) {
  const data: StackedPoint[] = series.points.map((point) => ({
    x: point.bucket,
    succeeded: point.succeeded,
    failed: point.failed,
  }))
  return (
    <TrendChart
      data={data}
      series={SERIES}
      formatValue={(value) => formatNumber(value)}
      formatXTick={(iso) => tick(iso, series.bucket)}
      ariaLabel={`Sessions per ${series.bucket}, succeeded and failed`}
      height={90}
      showYAxis
      yTickCount={3}
    />
  )
}
