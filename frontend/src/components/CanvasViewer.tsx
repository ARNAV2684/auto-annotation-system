import type Konva from 'konva'
import type { KonvaEventObject } from 'konva/lib/Node'
import { useLayoutEffect, useRef, useState } from 'react'
import { Image as KonvaImage, Layer, Line, Rect, Stage } from 'react-konva'
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

interface ViewTransform {
  scale: number
  panX: number
  panY: number
}

const MIN_SCALE = 0.05
const MAX_SCALE = 20
const ZOOM_FACTOR = 1.15
const MIN_BOX_PIXELS = 5

/**
 * Image canvas with mask overlay + zoom / pan + point/box prompts.
 *
 * Stage transform model
 * ---------------------
 * The Konva Stage carries the user's view transform (``scale`` + ``panX/Y``).
 * Inside the stage, the image is drawn at ``(0, 0)`` in its natural pixel size,
 * and every polygon is drawn at raw image-pixel coordinates. ``stage.getRelative
 * PointerPosition()`` then returns the cursor *in image-pixel coordinates* with
 * zero math from us — the cleanest possible coordinate story.
 *
 * Interactions
 * ------------
 * * Mouse wheel  → zoom toward the cursor (the world point under the cursor
 *   stays under the cursor).
 * * Middle-mouse drag, or Shift+drag  → pan.
 * * Left click in **point** mode → point-prompt at the cursor's image pixel.
 * * Left drag in **box** mode → box-prompt; the preview rectangle is drawn in
 *   image-pixel coords and uses ``strokeScaleEnabled={false}`` so the stroke
 *   stays crisp at any zoom level.
 * * Image change → automatically re-fit to the container (Reset button does the
 *   same on demand).
 *
 * Resize handling
 * ---------------
 * A ``ResizeObserver`` watches the container directly, so the Stage follows
 * sidebar drags, browser zoom, and any other reflow — not just window resize.
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
  const [view, setView] = useState<ViewTransform>({ scale: 1, panX: 0, panY: 0 })
  const [dragBox, setDragBox] = useState<DragBox | null>(null)
  // Cursor in image-pixel coords (for the crosshair guides) and the annotation
  // currently under the cursor (for hover highlight).
  const [cursorImg, setCursorImg] = useState<{ x: number; y: number } | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [panOrigin, setPanOrigin] = useState<{
    cursorX: number
    cursorY: number
    panX: number
    panY: number
  } | null>(null)
  // Track image+tool changes so we can reset transient state without an effect.
  const [prevImageId, setPrevImageId] = useState<string | null>(null)
  const [prevTool, setPrevTool] = useState<Tool>(tool)

  const imageElement = useImageElement(`/api/images/${image.image_id}/file`)
  const { data: annotations } = useAnnotations(image.image_id)
  const inferPoint = useInferPoint(image.image_id)
  const inferBox = useInferBox(image.image_id)

  // ResizeObserver: catches sidebar drags, browser zoom, and any container reflow.
  useLayoutEffect(() => {
    const node = containerRef.current
    if (!node) return
    const observer = new ResizeObserver(() => {
      // Async (browser-driven) -> allowed by react-hooks/set-state-in-effect.
      setStageSize({ width: node.clientWidth, height: node.clientHeight })
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  // Compute the "fit to container" transform — what Reset returns to.
  const fitTransform = (): ViewTransform => {
    if (!imageElement) return { scale: 1, panX: 0, panY: 0 }
    const scale = Math.min(
      stageSize.width / imageElement.naturalWidth,
      stageSize.height / imageElement.naturalHeight,
    )
    return {
      scale,
      panX: (stageSize.width - imageElement.naturalWidth * scale) / 2,
      panY: (stageSize.height - imageElement.naturalHeight * scale) / 2,
    }
  }

  // Derive-during-render: on image change, snap view to fit; reset drag state.
  if (prevImageId !== image.image_id && imageElement) {
    setPrevImageId(image.image_id)
    setView(fitTransform())
    setDragBox(null)
    setPanOrigin(null)
    setHoveredId(null)
    setCursorImg(null)
  }
  if (prevTool !== tool) {
    setPrevTool(tool)
    setDragBox(null)
  }

  // Cursor: pan trumps tool style.
  const cursorClass =
    panOrigin !== null
      ? 'cursor-grabbing'
      : tool === 'box'
        ? 'cursor-crosshair'
        : 'cursor-crosshair'

  // ─── interactions ──────────────────────────────────────────────────────

  const handleWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()
    const stage = e.target.getStage()
    if (!stage) return
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    const oldScale = view.scale
    const worldX = (pointer.x - view.panX) / oldScale
    const worldY = (pointer.y - view.panY) / oldScale
    const direction = e.evt.deltaY > 0 ? -1 : 1
    const newScale = clamp(
      direction > 0 ? oldScale * ZOOM_FACTOR : oldScale / ZOOM_FACTOR,
      MIN_SCALE,
      MAX_SCALE,
    )
    setView({
      scale: newScale,
      panX: pointer.x - worldX * newScale,
      panY: pointer.y - worldY * newScale,
    })
  }

  const isPanModifier = (evt: MouseEvent): boolean => evt.shiftKey || evt.button === 1

  const handleMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    if (!imageElement) return
    const stage = e.target.getStage()
    const screen = stage?.getPointerPosition()
    if (!screen) return

    if (isPanModifier(e.evt)) {
      setPanOrigin({ cursorX: screen.x, cursorY: screen.y, panX: view.panX, panY: view.panY })
      return
    }

    if (tool === 'box') {
      const img = stage!.getRelativePointerPosition()
      if (!img) return
      setDragBox({ x1: img.x, y1: img.y, x2: img.x, y2: img.y })
    }
  }

  const handleMouseMove = (e: KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage()
    const img = stage?.getRelativePointerPosition()
    if (img) setCursorImg({ x: img.x, y: img.y })

    if (panOrigin) {
      const screen = stage?.getPointerPosition()
      if (!screen) return
      setView({
        ...view,
        panX: panOrigin.panX + (screen.x - panOrigin.cursorX),
        panY: panOrigin.panY + (screen.y - panOrigin.cursorY),
      })
      return
    }
    if (dragBox && tool === 'box') {
      if (!img) return
      setDragBox({ ...dragBox, x2: img.x, y2: img.y })
    }
  }

  // Clear the crosshair when the cursor leaves the canvas entirely.
  const handleMouseLeave = () => setCursorImg(null)

  const handleMouseUp = () => {
    if (panOrigin) {
      setPanOrigin(null)
      return
    }
    if (dragBox && tool === 'box') {
      const x1 = Math.min(dragBox.x1, dragBox.x2)
      const y1 = Math.min(dragBox.y1, dragBox.y2)
      const x2 = Math.max(dragBox.x1, dragBox.x2)
      const y2 = Math.max(dragBox.y1, dragBox.y2)
      setDragBox(null)
      if (!imageElement) return
      if (x2 - x1 < MIN_BOX_PIXELS || y2 - y1 < MIN_BOX_PIXELS) return
      inferBox.mutate({
        x1: clamp(x1, 0, imageElement.naturalWidth),
        y1: clamp(y1, 0, imageElement.naturalHeight),
        x2: clamp(x2, 0, imageElement.naturalWidth),
        y2: clamp(y2, 0, imageElement.naturalHeight),
      })
    }
  }

  const handleClick = (e: KonvaEventObject<MouseEvent>) => {
    if (tool !== 'point' || !imageElement) return
    // Ignore clicks that started as pans.
    if (panOrigin !== null) return
    // Let MaskOverlay handle its own clicks; we only fire when the background was hit.
    if (e.target !== e.currentTarget && e.target.getClassName() !== 'Image') return
    const img = e.target.getStage()?.getRelativePointerPosition()
    if (!img) return
    if (
      img.x < 0 ||
      img.y < 0 ||
      img.x >= imageElement.naturalWidth ||
      img.y >= imageElement.naturalHeight
    ) {
      return
    }
    inferPoint.mutate({ x: img.x, y: img.y, label: 1 })
  }

  // ─── overlay UI: zoom % + Reset + hint ─────────────────────────────────

  const handleReset = () => setView(fitTransform())
  const handleZoomIn = () =>
    setView({ ...view, scale: clamp(view.scale * ZOOM_FACTOR, MIN_SCALE, MAX_SCALE) })
  const handleZoomOut = () =>
    setView({ ...view, scale: clamp(view.scale / ZOOM_FACTOR, MIN_SCALE, MAX_SCALE) })

  const busy = inferPoint.isPending || inferBox.isPending

  // Crosshair guides follow the cursor while it is over the image and we are not
  // mid-pan, so the user can line up a click/box precisely.
  const imgW = imageElement?.naturalWidth ?? 0
  const imgH = imageElement?.naturalHeight ?? 0
  const showCrosshair =
    !!imageElement &&
    !!cursorImg &&
    !panOrigin &&
    cursorImg.x >= 0 &&
    cursorImg.x <= imgW &&
    cursorImg.y >= 0 &&
    cursorImg.y <= imgH

  return (
    <div ref={containerRef} className={`relative h-full w-full bg-slate-100 ${cursorClass}`}>
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        scaleX={view.scale}
        scaleY={view.scale}
        x={view.panX}
        y={view.panY}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
        <Layer>
          {imageElement && (
            <KonvaImage
              image={imageElement}
              x={0}
              y={0}
              width={imageElement.naturalWidth}
              height={imageElement.naturalHeight}
              listening
            />
          )}
          {annotations && (
            <MaskOverlay
              annotations={annotations}
              selectedId={selectedAnnotationId}
              hoveredId={hoveredId}
              onSelect={onSelectAnnotation}
              onHover={setHoveredId}
            />
          )}
          {dragBox && tool === 'box' && (
            <Rect
              x={Math.min(dragBox.x1, dragBox.x2)}
              y={Math.min(dragBox.y1, dragBox.y2)}
              width={Math.abs(dragBox.x2 - dragBox.x1)}
              height={Math.abs(dragBox.y2 - dragBox.y1)}
              stroke="rgba(14, 165, 233, 0.95)"
              strokeWidth={1.5 / view.scale}
              fill="rgba(14, 165, 233, 0.15)"
              dash={[6 / view.scale, 6 / view.scale]}
              listening={false}
            />
          )}
          {showCrosshair && cursorImg && (
            <>
              <Line
                points={[cursorImg.x, 0, cursorImg.x, imgH]}
                stroke="rgba(14, 165, 233, 0.5)"
                strokeWidth={1}
                dash={[4, 4]}
                strokeScaleEnabled={false}
                listening={false}
              />
              <Line
                points={[0, cursorImg.y, imgW, cursorImg.y]}
                stroke="rgba(14, 165, 233, 0.5)"
                strokeWidth={1}
                dash={[4, 4]}
                strokeScaleEnabled={false}
                listening={false}
              />
            </>
          )}
        </Layer>
      </Stage>

      {/* HTML overlay: zoom controls + hint. Outside the Stage so transforms don't touch it. */}
      <div className="pointer-events-none absolute inset-0">
        <div className="pointer-events-auto absolute right-3 bottom-3 flex items-center gap-1 rounded-lg bg-white/95 px-2 py-1 text-sm shadow-md ring-1 ring-slate-200">
          <button
            type="button"
            onClick={handleZoomOut}
            className="grid h-7 w-7 place-items-center rounded text-slate-600 hover:bg-slate-100"
            title="Zoom out (mouse wheel)"
          >
            −
          </button>
          <span className="min-w-12 text-center font-mono text-xs text-slate-700">
            {Math.round(view.scale * 100)}%
          </span>
          <button
            type="button"
            onClick={handleZoomIn}
            className="grid h-7 w-7 place-items-center rounded text-slate-600 hover:bg-slate-100"
            title="Zoom in (mouse wheel)"
          >
            +
          </button>
          <span className="mx-1 h-4 w-px bg-slate-200" />
          <button
            type="button"
            onClick={handleReset}
            className="rounded px-2 py-0.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
            title="Reset view (fit image)"
          >
            Reset
          </button>
        </div>
        <div className="absolute bottom-3 left-3 rounded bg-white/85 px-2 py-1 text-xs text-slate-600 shadow ring-1 ring-slate-200">
          wheel = zoom · shift-drag or middle-drag = pan
        </div>
      </div>

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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}
