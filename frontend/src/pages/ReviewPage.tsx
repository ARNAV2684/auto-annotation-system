import { useEffect, useMemo, useState } from 'react'
import {
  useAnnotations,
  useDeleteAnnotation,
  useEmbed,
  useImages,
  useInferPoints,
  usePatchAnnotation,
} from '../api/hooks'
import type { PointSpec, Project } from '../api/types'
import { AutoDetectPanel } from '../components/AutoDetectPanel'
import { CanvasViewer, type Tool } from '../components/CanvasViewer'
import { DetectionList } from '../components/DetectionList'
import { ImageList } from '../components/ImageList'
import { ShortcutHelpOverlay } from '../components/ShortcutHelpOverlay'
import { SmartSelectPanel } from '../components/SmartSelectPanel'
import { ToolPalette } from '../components/ToolPalette'
import { useKeyboardShortcuts, type Shortcut } from '../components/useKeyboardShortcuts'

interface ReviewPageProps {
  /** Active project (UI preset). null while projects are loading. */
  project: Project | null
  onBack: () => void
  onOpenExport: () => void
}

/**
 * Three-column layout: left = image thumbnails, centre = canvas, right = controls
 * (auto-detect, detection list, export).
 *
 * Owns three pieces of UI state:
 *   - currentImageId (which image is being reviewed)
 *   - currentTool ('point' | 'box')
 *   - selectedAnnotationId (highlighted mask)
 *
 * Polish: as soon as the user opens an image, we pre-compute its SAM embedding
 * so the first point/box prompt feels instant instead of paying the ~0.5s
 * `set_image` cost on first click.
 */
export function ReviewPage({ project, onBack, onOpenExport }: ReviewPageProps) {
  const classes = project?.classes ?? []
  const { data: images } = useImages()
  const embed = useEmbed()
  const [currentImageId, setCurrentImageId] = useState<string | null>(null)
  const [currentTool, setCurrentTool] = useState<Tool>('point')
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null)
  const [helpOpen, setHelpOpen] = useState(false)
  // Smart Select state, owned here so the canvas and the side panel share it.
  const [smartPoints, setSmartPoints] = useState<PointSpec[]>([])
  const [simplify, setSimplify] = useState(2)
  // Tracks the previous image id / tool so we can reset transient state on change
  // (React 19's "derive state during render" pattern - no effect needed).
  const [prevImageId, setPrevImageId] = useState<string | null>(null)
  const [prevTool, setPrevTool] = useState<Tool>('point')

  // Auto-select the first image as soon as the list arrives (also during render
  // - no effect needed; the next render will see the new currentImageId).
  if (!currentImageId && images && images.length > 0) {
    setCurrentImageId(images[0].image_id)
  }
  if (prevImageId !== currentImageId) {
    setPrevImageId(currentImageId)
    setSelectedAnnotationId(null)
    setSmartPoints([])
  }
  if (prevTool !== currentTool) {
    setPrevTool(currentTool)
    if (currentTool !== 'smart') setSmartPoints([])
  }

  const inferPoints = useInferPoints(currentImageId ?? '')
  const finishSmart = () => {
    if (smartPoints.length === 0 || !currentImageId) return
    inferPoints.mutate({ points: smartPoints, simplify }, { onSuccess: () => setSmartPoints([]) })
  }
  const cancelSmart = () => setSmartPoints([])

  // Enter = finish, Esc = cancel while building a Smart Select (skip in inputs).
  useEffect(() => {
    if (currentTool !== 'smart' || smartPoints.length === 0) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (e.key === 'Enter') {
        e.preventDefault()
        if (currentImageId) {
          inferPoints.mutate(
            { points: smartPoints, simplify },
            { onSuccess: () => setSmartPoints([]) },
          )
        }
      } else if (e.key === 'Escape') {
        e.preventDefault()
        setSmartPoints([])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [currentTool, smartPoints, simplify, currentImageId, inferPoints])

  // Pre-warm SAM's image embedding whenever the user opens an image. Fires-
  // and-forgets the side effect (mutateAsync result is ignored).
  const embedMutate = embed.mutate
  useEffect(() => {
    if (currentImageId) {
      embedMutate(currentImageId)
    }
  }, [currentImageId, embedMutate])

  const currentImage = images?.find((i) => i.image_id === currentImageId) ?? null
  const { data: annotations } = useAnnotations(currentImageId)
  const patch = usePatchAnnotation()
  const del = useDeleteAnnotation()

  const shortcuts = useMemo<Shortcut[]>(() => {
    const items = images ?? []
    const anns = annotations ?? []
    const currentIdx = items.findIndex((i) => i.image_id === currentImageId)
    const selectedAnn = anns.find((a) => a.annotation_id === selectedAnnotationId) ?? null

    const stepImage = (delta: number) => {
      if (items.length === 0 || currentIdx < 0) return
      const next = (currentIdx + delta + items.length) % items.length
      setCurrentImageId(items[next].image_id)
    }
    const stepAnnotation = (delta: number) => {
      if (anns.length === 0) return
      const idx = selectedAnnotationId
        ? anns.findIndex((a) => a.annotation_id === selectedAnnotationId)
        : -1
      const next = idx < 0 ? 0 : (idx + delta + anns.length) % anns.length
      setSelectedAnnotationId(anns[next].annotation_id)
    }
    const setApproved = (approved: boolean | null) => {
      if (!selectedAnn) return
      patch.mutate({
        annotationId: selectedAnn.annotation_id,
        patch: { approved },
      })
    }
    const deleteSelected = () => {
      if (!selectedAnn) return
      del.mutate({
        annotationId: selectedAnn.annotation_id,
        imageId: selectedAnn.image_id,
      })
      setSelectedAnnotationId(null)
    }

    return [
      {
        combo: 'Shift+?',
        label: 'Show this help',
        group: 'General',
        handler: () => setHelpOpen(true),
      },
      {
        combo: 'Escape',
        label: 'Clear selection / close overlay',
        group: 'General',
        allowInInputs: true,
        handler: () => setSelectedAnnotationId(null),
      },
      {
        combo: 'Shift+A',
        label: 'Approve selected detection',
        group: 'Review',
        handler: () => setApproved(true),
      },
      {
        combo: 'Shift+R',
        label: 'Reject selected detection',
        group: 'Review',
        handler: () => setApproved(false),
      },
      {
        combo: 'Shift+D',
        label: 'Delete selected detection',
        group: 'Review',
        handler: deleteSelected,
      },
      {
        combo: 'Shift+J',
        label: 'Next detection',
        group: 'Review',
        handler: () => stepAnnotation(1),
      },
      {
        combo: 'Shift+K',
        label: 'Previous detection',
        group: 'Review',
        handler: () => stepAnnotation(-1),
      },
      {
        combo: 'Shift+N',
        label: 'Next image',
        group: 'Navigation',
        handler: () => stepImage(1),
      },
      {
        combo: 'Shift+P',
        label: 'Previous image',
        group: 'Navigation',
        handler: () => stepImage(-1),
      },
      {
        combo: 'Shift+S',
        label: 'Smart Select tool',
        group: 'Tools',
        handler: () => setCurrentTool('smart'),
      },
      {
        combo: 'Shift+T',
        label: 'Point (target) tool',
        group: 'Tools',
        handler: () => setCurrentTool('point'),
      },
      {
        combo: 'Shift+B',
        label: 'Box tool',
        group: 'Tools',
        handler: () => setCurrentTool('box'),
      },
      {
        combo: 'Shift+E',
        label: 'Open export dialog',
        group: 'General',
        handler: onOpenExport,
      },
    ]
  }, [images, annotations, currentImageId, selectedAnnotationId, patch, del, onOpenExport])

  // Disable global shortcuts while the help overlay is open — it owns its own
  // dismiss listener so any key closes it instead of triggering a page action.
  useKeyboardShortcuts(shortcuts, !helpOpen)

  return (
    // grid with sensible minimums: at narrow widths the page gets a horizontal
    // scrollbar instead of crushing the canvas column.
    <div className="grid h-full min-h-0 min-w-[900px] grid-cols-[240px_minmax(420px,1fr)_320px] overflow-auto">
      {/* Left: image list */}
      <aside className="flex min-h-0 min-w-0 flex-col border-r border-slate-200 bg-white">
        <div className="flex flex-shrink-0 items-center justify-between border-b border-slate-200 px-3 py-2">
          <span className="text-sm font-semibold text-slate-700">Images</span>
          <button
            type="button"
            onClick={onBack}
            className="text-xs text-slate-500 hover:text-slate-800"
          >
            ← upload
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          <ImageList selectedId={currentImageId} onSelect={setCurrentImageId} />
        </div>
      </aside>

      {/* Centre: canvas + tool palette */}
      <section className="relative flex min-h-0 min-w-0 flex-col bg-slate-100">
        <div className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-2">
          <span className="min-w-0 flex-1 truncate text-sm font-medium text-slate-700">
            {currentImage ? currentImage.file_name : 'No image selected'}
          </span>
          <ToolPalette current={currentTool} onChange={setCurrentTool} />
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            title="Shortcuts (Shift+?)"
            className="rounded border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-100"
          >
            ?
          </button>
        </div>
        <div className="relative min-h-0 flex-1 overflow-hidden">
          {currentImage ? (
            <CanvasViewer
              image={currentImage}
              tool={currentTool}
              selectedAnnotationId={selectedAnnotationId}
              onSelectAnnotation={setSelectedAnnotationId}
              smartPoints={smartPoints}
              onAddSmartPoint={(p) => setSmartPoints((prev) => [...prev, p])}
              simplify={simplify}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-slate-400">
              Select an image on the left.
            </div>
          )}
        </div>
      </section>

      {/* Right: detect + detections + export */}
      <aside className="flex min-h-0 min-w-0 flex-col gap-4 border-l border-slate-200 bg-slate-50 p-4">
        {currentImage && currentTool === 'smart' && (
          <div className="flex-shrink-0">
            <SmartSelectPanel
              pointCount={smartPoints.length}
              simplify={simplify}
              onSimplifyChange={setSimplify}
              onFinish={finishSmart}
              onCancel={cancelSmart}
              busy={inferPoints.isPending}
            />
          </div>
        )}
        {currentImage && (
          <div className="flex-shrink-0">
            <AutoDetectPanel imageId={currentImage.image_id} project={project} />
          </div>
        )}
        <div className="min-h-0 flex-1">
          {currentImage && (
            <DetectionList
              imageId={currentImage.image_id}
              classes={classes}
              selectedAnnotationId={selectedAnnotationId}
              onSelectAnnotation={setSelectedAnnotationId}
            />
          )}
        </div>
        <button
          type="button"
          onClick={onOpenExport}
          className="w-full flex-shrink-0 rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow transition hover:bg-slate-700"
        >
          Export annotations →
        </button>
      </aside>

      <ShortcutHelpOverlay
        open={helpOpen}
        shortcuts={shortcuts}
        onClose={() => setHelpOpen(false)}
      />
    </div>
  )
}
