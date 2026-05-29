import { useAnnotations } from '../api/hooks'
import { DetectionCard } from './DetectionCard'

interface DetectionListProps {
  imageId: string
  classes: string[]
  selectedAnnotationId: string | null
  onSelectAnnotation: (id: string) => void
}

/** Right-sidebar list of every annotation for the current image. */
export function DetectionList({
  imageId,
  classes,
  selectedAnnotationId,
  onSelectAnnotation,
}: DetectionListProps) {
  const { data: annotations, isLoading } = useAnnotations(imageId)

  if (isLoading) {
    return <div className="text-sm text-slate-500">Loading detections…</div>
  }

  const items = annotations ?? []
  const summary = `${items.length} detection${items.length === 1 ? '' : 's'}`

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-slate-700">Detections</span>
        <span className="text-xs text-slate-500">{summary}</span>
      </div>
      {items.length === 0 ? (
        <p className="rounded border border-dashed border-slate-300 bg-white p-3 text-sm text-slate-500">
          No detections yet. Click to add a point, drag to add a box, or run auto-detect.
        </p>
      ) : (
        <ul className="flex-1 space-y-2 overflow-auto pr-1">
          {items.map((ann) => (
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
