import { Line } from 'react-konva'
import type { Annotation } from '../api/types'

interface MaskOverlayProps {
  annotations: Annotation[]
  selectedId: string | null
  /** Current Stage scale - used to keep stroke widths constant in screen pixels. */
  viewScale: number
  onSelect: (annotationId: string) => void
}

// Colour code:
//   yellow = pending review
//   green  = approved
//   red    = rejected
function fillFor(approved: Annotation['approved'], highlighted: boolean): string {
  const alpha = highlighted ? 0.55 : 0.35
  if (approved === true) return `rgba(16, 185, 129, ${alpha})` // emerald
  if (approved === false) return `rgba(239, 68, 68, ${alpha})` // red
  return `rgba(250, 204, 21, ${alpha})` // amber / pending
}

function strokeFor(approved: Annotation['approved'], highlighted: boolean): string {
  const alpha = highlighted ? 1 : 0.85
  if (approved === true) return `rgba(5, 150, 105, ${alpha})`
  if (approved === false) return `rgba(220, 38, 38, ${alpha})`
  return `rgba(202, 138, 4, ${alpha})`
}

/**
 * Renders one Konva ``Line`` per polygon in raw image-pixel coords. The Stage
 * carries the view scale, so we only divide stroke widths by ``viewScale`` to
 * keep them constant in screen pixels regardless of zoom.
 */
export function MaskOverlay({ annotations, selectedId, viewScale, onSelect }: MaskOverlayProps) {
  const baseStroke = 1.5 / viewScale
  const highlightStroke = 3 / viewScale
  return (
    <>
      {annotations.map((ann) =>
        ann.polygon.map((polygon, polyIdx) => {
          const highlighted = ann.annotation_id === selectedId
          return (
            <Line
              key={`${ann.annotation_id}-${polyIdx}`}
              points={polygon}
              closed
              fill={fillFor(ann.approved, highlighted)}
              stroke={strokeFor(ann.approved, highlighted)}
              strokeWidth={highlighted ? highlightStroke : baseStroke}
              strokeScaleEnabled={false}
              onClick={() => onSelect(ann.annotation_id)}
              onTap={() => onSelect(ann.annotation_id)}
              listening
            />
          )
        }),
      )}
    </>
  )
}
