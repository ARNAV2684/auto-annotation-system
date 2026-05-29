import { useStats } from '../api/hooks'

/** Compact live counter of images / detections / approved. Updates after every mutation. */
export function StatsWidget() {
  const { data, isLoading, isError } = useStats()

  if (isLoading) {
    return <span className="text-sm text-slate-400">loading stats…</span>
  }
  if (isError || !data) {
    return <span className="text-sm text-red-500">stats unavailable</span>
  }

  return (
    <div className="flex items-center gap-4 text-sm text-slate-700">
      <Stat label="images" value={data.num_images} />
      <Stat label="detections" value={data.num_annotations} />
      <Stat label="approved" value={data.num_approved} accent="text-emerald-600" />
      <Stat label="pending" value={data.num_pending} accent="text-amber-600" />
    </div>
  )
}

function Stat({
  label,
  value,
  accent = 'text-slate-900',
}: {
  label: string
  value: number
  accent?: string
}) {
  return (
    <span className="flex items-baseline gap-1">
      <span className={`font-semibold ${accent}`}>{value}</span>
      <span className="text-slate-500">{label}</span>
    </span>
  )
}
