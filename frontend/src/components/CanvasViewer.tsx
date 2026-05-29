import type Konva from 'konva'
import type { KonvaEventObject } from 'konva/lib/Node'
import { useLayoutEffect, useRef, useState } from 'react'
import { Group, Image as KonvaImage, Layer, Rect, Stage } from 'react-konva'
import { useAnnotations, useInferBox, useInferPoint } from '../api/hooks'
import type { ImageRecord } from '../api/types'
import { MaskOverlay } from './MaskOverlay'
import { useImageElement } from './useImageElement'

export type Tool = 'point' | 'box'

interface CanvasViewerProps {
  image: ImageRecord
  tool: Tool
  selectedAnnotationId: string | null
  onSelectAnnotation: (id: string | null) => void
}

interface DragBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

/**
 * Image canvas with mask overlay + point/box prompt interactions.
 *
 * - Image is fit to the container; we track ``scale`` so polygons (which are in
 *   image-pixel coordinates) render at the right size.
 * - Click in "point" mode -> useInferPoint mutation, in image-pixel coords.
 * - Drag in "box" mode -> shows a translucent rectangle; on release ->
 *   useInferBox mutation in image-pixel coords.
 */
export function CanvasViewer({
  image,
  tool,
  selectedAnnotationId,
  onSelectAnnotation,
}: CanvasViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<Konva.Stage>(null)
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 })
  const [dragBox, setDragBox] = useState<DragBox | null>(null)
  // Track image/tool changes so we can reset the in-progress drag without an effect.
  const [prevKey, setPrevKey] = useState<string>(`${image.image_id}/${tool}`)

  const imageElement = useImageElement(`/api/images/${image.image_id}/file`)
  const { data: annotations } = useAnnotations(image.image_id)
  const inferPoint = useInferPoint(image.image_id)
  const inferBox = useInferBox(image.image_id)

  // Derive-during-render reset (avoids react-hooks/set-state-in-effect on a
  // simple prop-change reset).
  const currentKey = `${image.image_id}/${tool}`
  if (prevKey !== currentKey) {
    setPrevKey(currentKey)
    setDragBox(null)
  }

  // Resize stage to fit container. The setState inside `measure` runs
  // asynchronously (from a window resize event) so the new lint rule allows it.
  useLayoutEffect(() => {
    const measure = () => {
      const node = containerRef.current
      if (!node) return
      setStageSize({ width: node.clientWidth, height: node.clientHeight })
    }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [])

  // Fit the image into the stage; same scale on both axes (no distortion).
  const scale = imageElement
    ? Math.min(
        stageSize.width / imageElement.naturalWidth,
        stageSize.height / imageElement.naturalHeight,
      )
    : 1
  const scaledWidth = imageElement ? imageElement.naturalWidth * scale : 0
  const scaledHeight = imageElement ? imageElement.naturalHeight * scale : 0
  const offsetX = (stageSize.width - scaledWidth) / 2
  const offsetY = (stageSize.height - scaledHeight) / 2

  const toImagePixel = (stageX: number, stageY: number): { x: number; y: number } => {
    return {
      x: Math.max(0, (stageX - offsetX) / scale),
      y: Math.max(0, (stageY - offsetY) / scale),
    }
  }

  const handleMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    if (tool !== 'box' || !imageElement) return
    const stage = e.target.getStage()
    const pointer = stage?.getPointerPosition()
    if (!pointer) return
    setDragBox({ x1: pointer.x, y1: pointer.y, x2: pointer.x, y2: pointer.y })
  }

  const handleMouseMove = (e: KonvaEventObject<MouseEvent>) => {
    if (!dragBox || tool !== 'box') return
    const pointer = e.target.getStage()?.getPointerPosition()
    if (!pointer) return
    setDragBox({ ...dragBox, x2: pointer.x, y2: pointer.y })
  }

  const handleMouseUp = () => {
    if (!dragBox || tool !== 'box') return
    const x1Img = Math.min(dragBox.x1, dragBox.x2)
    const y1Img = Math.min(dragBox.y1, dragBox.y2)
    const x2Img = Math.max(dragBox.x1, dragBox.x2)
    const y2Img = Math.max(dragBox.y1, dragBox.y2)
    const tl = toImagePixel(x1Img, y1Img)
    const br = toImagePixel(x2Img, y2Img)
    const pixelW = br.x - tl.x
    const pixelH = br.y - tl.y
    setDragBox(null)
    if (pixelW < 5 || pixelH < 5) return // too small / mis-click
    inferBox.mutate({ x1: tl.x, y1: tl.y, x2: br.x, y2: br.y })
  }

  const handleClick = (e: KonvaEventObject<MouseEvent>) => {
    if (tool !== 'point' || !imageElement) return
    // If they clicked a mask (Konva propagates through), let MaskOverlay handle it.
    if (e.target !== e.currentTarget && e.target.getClassName() !== 'Image') return
    const pointer = e.target.getStage()?.getPointerPosition()
    if (!pointer) return
    const { x, y } = toImagePixel(pointer.x, pointer.y)
    // Clicking outside the image area is a no-op.
    if (x <= 0 || y <= 0 || x >= imageElement.naturalWidth || y >= imageElement.naturalHeight)
      return
    inferPoint.mutate({ x, y, label: 1 })
  }

  const busy = inferPoint.isPending || inferBox.isPending
  const cursor = tool === 'box' ? 'crosshair' : 'pointer'

  return (
    <div ref={containerRef} className="relative h-full w-full bg-slate-100" style={{ cursor }}>
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
      >
        <Layer>
          {imageElement && (
            <KonvaImage
              image={imageElement}
              x={offsetX}
              y={offsetY}
              width={scaledWidth}
              height={scaledHeight}
              listening
            />
          )}
          {annotations && (
            // Translate so polygons (image-pixel coords scaled to stage) line up
            // with the centred image. Children only need to multiply by `scale`.
            <Group x={offsetX} y={offsetY}>
              <MaskOverlay
                annotations={annotations}
                selectedId={selectedAnnotationId}
                scale={scale}
                onSelect={onSelectAnnotation}
              />
            </Group>
          )}
          {dragBox && tool === 'box' && (
            <Rect
              x={Math.min(dragBox.x1, dragBox.x2)}
              y={Math.min(dragBox.y1, dragBox.y2)}
              width={Math.abs(dragBox.x2 - dragBox.x1)}
              height={Math.abs(dragBox.y2 - dragBox.y1)}
              stroke="rgba(14, 165, 233, 0.9)"
              strokeWidth={1.5}
              fill="rgba(14, 165, 233, 0.15)"
              dash={[4, 4]}
              listening={false}
            />
          )}
        </Layer>
        {/* Translate the offset so polygons (image-pixel coords) appear at the right place. */}
      </Stage>
      {busy && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-white/40">
          <span className="rounded bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow">
            Segmenting…
          </span>
        </div>
      )}
    </div>
  )
}
