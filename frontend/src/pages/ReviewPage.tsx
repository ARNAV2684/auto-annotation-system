import { useEffect, useState } from 'react'
import { useEmbed, useImages } from '../api/hooks'
import type { Project } from '../api/types'
import { AutoDetectPanel } from '../components/AutoDetectPanel'
import { CanvasViewer, type Tool } from '../components/CanvasViewer'
import { DetectionList } from '../components/DetectionList'
import { ImageList } from '../components/ImageList'
import { ToolPalette } from '../components/ToolPalette'

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
  // Tracks the previous image id so we can reset selection when it changes
  // (React 19's "derive state during render" pattern - no effect needed).
  const [prevImageId, setPrevImageId] = useState<string | null>(null)

  // Auto-select the first image as soon as the list arrives (also during render
  // - no effect needed; the next render will see the new currentImageId).
  if (!currentImageId && images && images.length > 0) {
    setCurrentImageId(images[0].image_id)
  }
  if (prevImageId !== currentImageId) {
    setPrevImageId(currentImageId)
    setSelectedAnnotationId(null)
  }

  // Pre-warm SAM's image embedding whenever the user opens an image. Fires-
  // and-forgets the side effect (mutateAsync result is ignored).
  const embedMutate = embed.mutate
  useEffect(() => {
    if (currentImageId) {
      embedMutate(currentImageId)
    }
  }, [currentImageId, embedMutate])

  const currentImage = images?.find((i) => i.image_id === currentImageId) ?? null

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
        </div>
        <div className="relative min-h-0 flex-1 overflow-hidden">
          {currentImage ? (
            <CanvasViewer
              image={currentImage}
              tool={currentTool}
              selectedAnnotationId={selectedAnnotationId}
              onSelectAnnotation={setSelectedAnnotationId}
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
    </div>
  )
}
