import { useState } from 'react'
import { useDeleteAnnotation, usePatchAnnotation } from '../api/hooks'
import type { Annotation } from '../api/types'
import { InlineClassPopup } from './InlineClassPopup'

const ADD_NEW = '__add_new__'

interface DetectionCardProps {
  annotation: Annotation
  classes: string[]
  selected: boolean
  onSelect: () => void
}

/** One row in the detections panel: source/score, approve/reject/delete, label dropdown. */
export function DetectionCard({ annotation, classes, selected, onSelect }: DetectionCardProps) {
  const patch = usePatchAnnotation()
  const del = useDeleteAnnotation()
  const [popupOpen, setPopupOpen] = useState(false)

  const setApproved = (approved: boolean | null) =>
    patch.mutate({ annotationId: annotation.annotation_id, patch: { approved } })

  const setLabel = (label: string) =>
    patch.mutate({ annotationId: annotation.annotation_id, patch: { label } })

  const remove = () =>
    del.mutate({ annotationId: annotation.annotation_id, imageId: annotation.image_id })

  const stateClasses =
    annotation.approved === true
      ? 'border-emerald-300'
      : annotation.approved === false
        ? 'border-red-300 opacity-70'
        : 'border-amber-300'

  const selectedClasses = selected ? 'ring-2 ring-sky-400' : ''

  return (
    <li
      onClick={onSelect}
      className={`cursor-pointer rounded border bg-white p-2 shadow-sm transition ${stateClasses} ${selectedClasses}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <select
          value={classes.includes(annotation.label) ? annotation.label : ''}
          onChange={(e) => {
            e.stopPropagation()
            if (e.target.value === ADD_NEW) {
              setPopupOpen(true)
              return
            }
            setLabel(e.target.value)
          }}
          onClick={(e) => e.stopPropagation()}
          className="min-w-0 flex-1 truncate rounded border border-slate-200 px-1 py-0.5 text-sm text-slate-800"
        >
          {!classes.includes(annotation.label) && (
            <option value="" disabled>
              {annotation.label || 'choose…'} (not in current project)
            </option>
          )}
          {classes.map((cls) => (
            <option key={cls} value={cls}>
              {cls}
            </option>
          ))}
          <option value={ADD_NEW}>+ new class…</option>
        </select>
        <span className="font-mono text-xs text-slate-500">
          {(annotation.score * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mt-1.5 flex items-center justify-between">
        <span className="text-xs text-slate-500">{labelForSource(annotation.source)}</span>
        <div className="flex items-center gap-1">
          <ActionButton
            label="✓"
            title="Approve"
            active={annotation.approved === true}
            activeClass="bg-emerald-500 text-white"
            onClick={(e) => {
              e.stopPropagation()
              setApproved(annotation.approved === true ? null : true)
            }}
          />
          <ActionButton
            label="✗"
            title="Reject"
            active={annotation.approved === false}
            activeClass="bg-red-500 text-white"
            onClick={(e) => {
              e.stopPropagation()
              setApproved(annotation.approved === false ? null : false)
            }}
          />
          <ActionButton
            label="🗑"
            title="Delete"
            active={false}
            activeClass=""
            onClick={(e) => {
              e.stopPropagation()
              remove()
            }}
          />
        </div>
      </div>
      <InlineClassPopup
        open={popupOpen}
        onClose={() => setPopupOpen(false)}
        onRegistered={(name) => setLabel(name)}
      />
    </li>
  )
}

function labelForSource(source: Annotation['source']): string {
  if (source === 'auto') return 'auto-detect'
  if (source === 'point') return 'point click'
  return 'box drag'
}

function ActionButton({
  label,
  title,
  active,
  activeClass,
  onClick,
}: {
  label: string
  title: string
  active: boolean
  activeClass: string
  onClick: (e: React.MouseEvent) => void
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`h-7 w-7 rounded text-sm transition ${
        active ? activeClass : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
      }`}
    >
      {label}
    </button>
  )
}
