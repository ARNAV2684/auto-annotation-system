import { useMemo, useState } from 'react'
import { useAnnotations, useBulkAnnotations } from '../api/hooks'
import type { Annotation, BulkAction } from '../api/types'
import { DetectionCard } from './DetectionCard'

interface DetectionListProps {
  imageId: string
  classes: string[]
  selectedAnnotationId: string | null
  onSelectAnnotation: (id: string) => void
}

type StateFilter = 'all' | 'pending' | 'approved' | 'rejected'

const STATE_FILTERS: { key: StateFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'pending', label: 'Pending' },
  { key: 'approved', label: 'Approved' },
  { key: 'rejected', label: 'Rejected' },
]

function matchesStateFilter(ann: Annotation, filter: StateFilter): boolean {
  if (filter === 'all') return true
  if (filter === 'pending') return ann.approved === null
  if (filter === 'approved') return ann.approved === true
  return ann.approved === false
}

/** Right-sidebar list of every annotation for the current image. */
export function DetectionList({
  imageId,
  classes,
  selectedAnnotationId,
  onSelectAnnotation,
}: DetectionListProps) {
  const { data: annotations, isLoading } = useAnnotations(imageId)
  const bulk = useBulkAnnotations(imageId)
  const [stateFilter, setStateFilter] = useState<StateFilter>('all')
  // Score threshold is a percentage 0..100; only show annotations with score*100 >= threshold.
  const [minScorePct, setMinScorePct] = useState(0)

  const items = useMemo(() => annotations ?? [], [annotations])
  const visibleItems = useMemo(
    () =>
      items.filter(
        (a) => matchesStateFilter(a, stateFilter) && a.score * 100 >= minScorePct,
      ),
    [items, stateFilter, minScorePct],
  )

  const runBulk = (action: BulkAction) => {
    const ids = visibleItems.map((a) => a.annotation_id)
    if (ids.length === 0) return
    if (ids.length > 5) {
      const verb =
        action === 'approve' ? 'approve' : action === 'reject' ? 'reject' : 'delete'
      const ok = window.confirm(`${verb[0].toUpperCase() + verb.slice(1)} ${ids.length} annotations?`)
      if (!ok) return
    }
    bulk.mutate({ action, annotation_ids: ids })
  }

  if (isLoading) {
    return <div className="text-sm text-slate-500">Loading detections…</div>
  }

  const showingLabel =
    visibleItems.length === items.length
      ? `${items.length} detection${items.length === 1 ? '' : 's'}`
      : `showing ${visibleItems.length} of ${items.length}`

  const bulkDisabled = visibleItems.length === 0 || bulk.isPending

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-slate-700">Detections</span>
        <span className="text-xs text-slate-500">{showingLabel}</span>
      </div>

      {items.length > 0 && (
        <div className="mb-2 flex flex-shrink-0 flex-col gap-2 rounded border border-slate-200 bg-white p-2">
          <div className="flex flex-wrap gap-1">
            {STATE_FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                onClick={() => setStateFilter(f.key)}
                className={`rounded px-2 py-0.5 text-xs transition ${
                  stateFilter === f.key
                    ? 'bg-slate-800 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-600">
            <span className="w-16 flex-shrink-0">min score</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={minScorePct}
              onChange={(e) => setMinScorePct(Number(e.target.value))}
              className="flex-1"
            />
            <span className="w-9 text-right font-mono">{minScorePct}%</span>
          </label>
          <div className="flex gap-1">
            <BulkButton
              label="Approve all"
              tone="emerald"
              onClick={() => runBulk('approve')}
              disabled={bulkDisabled}
            />
            <BulkButton
              label="Reject all"
              tone="red"
              onClick={() => runBulk('reject')}
              disabled={bulkDisabled}
            />
            <BulkButton
              label="Delete all"
              tone="slate"
              onClick={() => runBulk('delete')}
              disabled={bulkDisabled}
            />
          </div>
        </div>
      )}

      {items.length === 0 ? (
        <p className="rounded border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
          No detections yet. Click to add a point, drag to add a box, or run auto-detect.
        </p>
      ) : visibleItems.length === 0 ? (
        <p className="rounded border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
          Nothing matches the current filter. Loosen the score threshold or pick a different state.
        </p>
      ) : (
        <ul className="flex-1 space-y-2 overflow-auto pr-1">
          {visibleItems.map((ann) => (
            <DetectionCard
              key={ann.annotation_id}
              annotation={ann}
              classes={classes}
              selected={ann.annotation_id === selectedAnnotationId}
              onSelect={() => onSelectAnnotation(ann.annotation_id)}
            />
          ))}
        </ul>
      )}
    </div>
  )
}

function BulkButton({
  label,
  tone,
  onClick,
  disabled,
}: {
  label: string
  tone: 'emerald' | 'red' | 'slate'
  onClick: () => void
  disabled: boolean
}) {
  const toneClasses =
    tone === 'emerald'
      ? 'bg-emerald-600 hover:bg-emerald-700'
      : tone === 'red'
        ? 'bg-red-600 hover:bg-red-700'
        : 'bg-slate-700 hover:bg-slate-800'
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex-1 rounded px-2 py-1 text-xs font-medium text-white transition disabled:cursor-not-allowed disabled:opacity-50 ${toneClasses}`}
    >
      {label}
    </button>
  )
}
