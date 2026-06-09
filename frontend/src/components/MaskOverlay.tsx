import { Line } from 'react-konva'
import type { Annotation } from '../api/types'
import { colorForLabel, rgba, type RGB } from '../utils/classColors'

interface MaskOverlayProps {
  annotations: Annotation[]
  selectedId: string | null
  hoveredId: string | null
  onSelect: (annotationId: string) => void
  onHover: (annotationId: string | null) => void
}

interface MaskStyle {
  fill: string
  stroke: string
  strokeWidth: number
  dash: number[] | undefined
}

/**
 * Compute a mask's style under "colour by class" (Option A):
 *   - **hue** comes from the class label, so every class is visually distinct;
 *   - **review state** rides on a separate channel so it never fights the hue:
 *       approved  = solid, more opaque
 *       pending   = solid, normal opacity
 *       rejected  = faded + dashed outline
 *   - hover brightens; selection thickens the outline.
 *
 * Stroke widths are screen pixels (``strokeScaleEnabled={false}``), so they stay
 * crisp at any zoom without dividing by the view scale.
 */
function maskStyle(
  approved: Annotation['approved'],
  base: RGB,
  selected: boolean,
  hovered: boolean,
): MaskStyle {
  let fillAlpha: number
  let strokeAlpha: number
  let dash: number[] | undefined
  if (approved === true) {
    fillAlpha = 0.42
    strokeAlpha = 1
    dash = undefined
  } else if (approved === false) {
    fillAlpha = 0.1
    strokeAlpha = 0.55
    dash = [6, 4] // dashed reads as "rejected"
  } else {
    fillAlpha = 0.3
    strokeAlpha = 0.9
    dash = undefined
  }
  if (hovered) fillAlpha = Math.min(fillAlpha + 0.15, 0.6)
  if (hovered || selected) strokeAlpha = 1
  const strokeWidth = selected ? 3.5 : hovered ? 2.5 : 1.5
  return { fill: rgba(base, fillAlpha), stroke: rgba(base, strokeAlpha), strokeWidth, dash }
}

/**
 * Renders one Konva ``Line`` per polygon in raw image-pixel coords, coloured by
 * class with review state on opacity/dashing. Reports hover so the canvas can
 * highlight the object under the cursor.
 */
export function MaskOverlay({
  annotations,
  selectedId,
  hoveredId,
  onSelect,
  onHover,
}: MaskOverlayProps) {
  return (
    <>
      {annotations.map((ann) => {
        const base = colorForLabel(ann.label)
        const selected = ann.annotation_id === selectedId
        const hovered = ann.annotation_id === hoveredId
        const style = maskStyle(ann.approved, base, selected, hovered)
        return ann.polygon.map((polygon, polyIdx) => (
          <Line
            key={`${ann.annotation_id}-${polyIdx}`}
            points={polygon}
            closed
            fill={style.fill}
            stroke={style.stroke}
            strokeWidth={style.strokeWidth}
            dash={style.dash}
            strokeScaleEnabled={false}
            onClick={() => onSelect(ann.annotation_id)}
            onTap={() => onSelect(ann.annotation_id)}
            onMouseEnter={() => onHover(ann.annotation_id)}
            onMouseLeave={() => onHover(null)}
            listening
          />
        ))
      })}
    </>
  )
}
