import type Konva from 'konva'
import type { KonvaEventObject } from 'konva/lib/Node'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Circle, Image as KonvaImage, Layer, Line, Rect, Stage } from 'react-konva'
import { useAnnotations, useCreatePolygon, useInferBox, useInferPoint } from '../api/hooks'
import type { Annotation, ImageRecord, Polygon, SmartObject } from '../api/types'
import { MaskOverlay } from './MaskOverlay'
import { useImageElement } from './useImageElement'

export type Tool = 'point' | 'box' | 'smart' | 'polygon'

interface CanvasViewerProps {
  image: ImageRecord
  tool: Tool
  selectedAnnotationId: string | null
  onSelectAnnotation: (id: string | null) => void
  /** Smart Select collected objects (owned by ReviewPage so the panel shares them). */
  smartObjects?: SmartObject[]
  onAddSmartObject?: (object: SmartObject) => void
  /** Fired with newly-created annotations so the parent can record undo history. */
  onAnnotationsCreated?: (created: Annotation[]) => void
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
  smartObjects = [],
  onAddSmartObject,
  onAnnotationsCreated,
}: CanvasViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<Konva.Stage>(null)
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 })
  const [view, setView] = useState<ViewTransform>({ scale: 1, panX: 0, panY: 0 })
  const [dragBox, setDragBox] = useState<DragBox | null>(null)
  // Smart Select ghost preview polygon for the object under the cursor (geometry
  // only; never persisted here).
  const [previewPolygon, setPreviewPolygon] = useState<Polygon[] | null>(null)
  const previewAbort = useRef<AbortController | null>(null)
  const lastPreviewAt = useRef(0)
  // Manual Polygon tool: vertices accumulated for the current draw.
  const [polygonPoints, setPolygonPoints] = useState<{ x: number; y: number }[]>([])
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
  const createPolygon = useCreatePolygon(image.image_id)

  // Smart Select preview: a cheap, non-persisting SAM call at the hovered point.
  // Stale requests are aborted so only the latest cursor position paints the ghost.
  const requestPreview = useCallback(
    (x: number, y: number) => {
      previewAbort.current?.abort()
      const controller = new AbortController()
      previewAbort.current = controller
      fetch(`/api/images/${image.image_id}/infer/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: [{ x, y, label: 1 }] }),
        signal: controller.signal,
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (data) setPreviewPolygon(data.polygon as Polygon[])
        })
        .catch(() => {
          /* aborted or failed — leave the previous preview in place */
        })
    },
    [image.image_id],
  )

  // Manual Polygon: commit (rasterise to a mask) on Enter / double-click; Esc cancels.
  const createPolygonMutate = createPolygon.mutate
  useEffect(() => {
    if (tool !== 'polygon' || polygonPoints.length === 0) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (e.key === 'Enter' && polygonPoints.length >= 3) {
        e.preventDefault()
        createPolygonMutate(
          { points: polygonPoints.map((p) => [p.x, p.y]) },
          {
            onSuccess: (ann) => {
              setPolygonPoints([])
              onAnnotationsCreated?.([ann])
            },
          },
        )
      } else if (e.key === 'Escape') {
        e.preventDefault()
        setPolygonPoints([])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [tool, polygonPoints, createPolygonMutate, onAnnotationsCreated])

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
    setPreviewPolygon(null)
    setPolygonPoints([])
  }
  if (prevTool !== tool) {
    setPrevTool(tool)
    setDragBox(null)
    setPreviewPolygon(null)
    setPolygonPoints([])
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
      return
    }
    // Smart Select hover preview: the ghost mask follows the cursor over each
    // object (always, so you can preview the next object right after collecting
    // one). Throttled to keep SAM calls reasonable.
    if (tool === 'smart' && img && imageElement) {
      const inBounds =
        img.x >= 0 &&
        img.y >= 0 &&
        img.x < imageElement.naturalWidth &&
        img.y < imageElement.naturalHeight
      const now = performance.now()
      if (inBounds && now - lastPreviewAt.current > 120) {
        lastPreviewAt.current = now
        requestPreview(img.x, img.y)
      }
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
      inferBox.mutate(
        {
          x1: clamp(x1, 0, imageElement.naturalWidth),
          y1: clamp(y1, 0, imageElement.naturalHeight),
          x2: clamp(x2, 0, imageElement.naturalWidth),
          y2: clamp(y2, 0, imageElement.naturalHeight),
        },
        { onSuccess: (ann) => onAnnotationsCreated?.([ann]) },
      )
    }
  }

  const handleClick = (e: KonvaEventObject<MouseEvent>) => {
    if (!imageElement) return
    // Ignore clicks that started as pans.
    if (panOrigin !== null) return
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
    if (tool === 'smart') {
      // Collect the object currently under the cursor (its previewed outline),
      // then keep hovering for the next one.
      onAddSmartObject?.({ x: img.x, y: img.y, polygon: previewPolygon ?? [] })
      return
    }
    if (tool === 'polygon') {
      setPolygonPoints((pts) => [...pts, { x: img.x, y: img.y }])
      return
    }
    if (tool === 'point') {
      // Let MaskOverlay handle its own clicks; only fire on a background hit.
      if (e.target !== e.currentTarget && e.target.getClassName() !== 'Image') return
      inferPoint.mutate(
        { x: img.x, y: img.y, label: 1 },
        { onSuccess: (anns) => onAnnotationsCreated?.(anns) },
      )
    }
  }

  // Double-click finishes a polygon (same as Enter).
  const handleDblClick = () => {
    if (tool === 'polygon' && polygonPoints.length >= 3) {
      createPolygon.mutate(
        { points: polygonPoints.map((p) => [p.x, p.y]) },
        {
          onSuccess: (ann) => {
            setPolygonPoints([])
            onAnnotationsCreated?.([ann])
          },
        },
      )
    }
  }

  // ─── overlay UI: zoom % + Reset + hint ─────────────────────────────────

  const handleReset = () => setView(fitTransform())
  const handleZoomIn = () =>
    setView({ ...view, scale: clamp(view.scale * ZOOM_FACTOR, MIN_SCALE, MAX_SCALE) })
  const handleZoomOut = () =>
    setView({ ...view, scale: clamp(view.scale / ZOOM_FACTOR, MIN_SCALE, MAX_SCALE) })

  const busy = inferPoint.isPending || inferBox.isPending || createPolygon.isPending

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
        onDblClick={handleDblClick}
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
              interactive={tool !== 'smart' && tool !== 'polygon'}
              onSelect={onSelectAnnotation}
              onHover={setHoveredId}
            />
          )}
          {tool === 'polygon' && polygonPoints.length > 0 && (
            <>
              <Line
                points={[
                  ...polygonPoints.flatMap((p) => [p.x, p.y]),
                  ...(cursorImg ? [cursorImg.x, cursorImg.y] : []),
                ]}
                closed={polygonPoints.length >= 3}
                stroke="rgba(217, 70, 239, 0.95)"
                strokeWidth={1.5}
                fill="rgba(217, 70, 239, 0.18)"
                dash={[5, 4]}
                strokeScaleEnabled={false}
                listening={false}
              />
              {polygonPoints.map((p, i) => (
                <Circle
                  key={`poly-pt-${i}`}
                  x={p.x}
                  y={p.y}
                  radius={4}
                  fill="#d946ef"
                  stroke="#ffffff"
                  strokeWidth={1.5}
                  strokeScaleEnabled={false}
                  listening={false}
                />
              ))}
            </>
          )}
          {/* Smart Select: collected objects (solid) + live hover ghost (dashed). */}
          {tool === 'smart' &&
            smartObjects.flatMap((obj, oi) =>
              obj.polygon.map((poly, pi) => (
                <Line
                  key={`collected-${oi}-${pi}`}
                  points={poly}
                  closed
                  fill="rgba(14, 165, 233, 0.35)"
                  stroke="rgba(2, 132, 199, 1)"
                  strokeWidth={2}
                  strokeScaleEnabled={false}
                  listening={false}
                />
              )),
            )}
          {tool === 'smart' &&
            smartObjects.map((obj, i) => (
              <Circle
                key={`collected-pt-${i}`}
                x={obj.x}
                y={obj.y}
                radius={4}
                fill="#0284c7"
                stroke="#ffffff"
                strokeWidth={1.5}
                strokeScaleEnabled={false}
                listening={false}
              />
            ))}
          {tool === 'smart' &&
            previewPolygon?.map((poly, i) => (
              <Line
                key={`preview-${i}`}
                points={poly}
                closed
                fill="rgba(14, 165, 233, 0.2)"
                stroke="rgba(14, 165, 233, 0.95)"
                strokeWidth={1.5}
                dash={[5, 4]}
                strokeScaleEnabled={false}
                listening={false}
              />
            ))}
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
