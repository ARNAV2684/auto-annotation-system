import { useEffect, useState } from 'react'
import { useEmbed, useImages } from '../api/hooks'
import { AutoDetectPanel } from '../components/AutoDetectPanel'
import { CanvasViewer, type Tool } from '../components/CanvasViewer'
import { DetectionList } from '../components/DetectionList'
import { ImageList } from '../components/ImageList'
import { ToolPalette } from '../components/ToolPalette'

interface ReviewPageProps {
  classes: string[]
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
export function ReviewPage({ classes, onBack, onOpenExport }: ReviewPageProps) {
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
    <div className="grid h-full grid-cols-[260px_minmax(0,1fr)_320px]">
      {/* Left: image list */}
      <aside className="flex flex-col border-r border-slate-200 bg-white">
        <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
          <span className="text-sm font-semibold text-slate-700">Images</span>
          <button
            type="button"
            onClick={onBack}
            className="text-xs text-slate-500 hover:text-slate-800"
          >
            ← upload
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          <ImageList selectedId={currentImageId} onSelect={setCurrentImageId} />
        </div>
      </aside>

      {/* Centre: canvas + tool palette */}
      <section className="relative flex flex-col bg-slate-100">
        <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2">
          <span className="truncate text-sm font-medium text-slate-700">
            {currentImage ? currentImage.file_name : 'No image selected'}
          </span>
          <ToolPalette current={currentTool} onChange={setCurrentTool} />
        </div>
        <div className="relative flex-1 overflow-hidden">
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
      <aside className="flex flex-col gap-4 border-l border-slate-200 bg-slate-50 p-4">
        {currentImage && <AutoDetectPanel imageId={currentImage.image_id} />}
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
          className="w-full rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow transition hover:bg-slate-700"
        >
          Export annotations →
        </button>
      </aside>
    </div>
  )
}
