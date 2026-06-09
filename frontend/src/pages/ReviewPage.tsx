import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  useAnnotations,
  useCopyAnnotations,
  useCreatePolygon,
  useDeleteAnnotation,
  useEmbed,
  useImages,
  useInferPoints,
  usePatchAnnotation,
  usePatchImage,
} from '../api/hooks'
import type { Annotation, PointSpec, Project } from '../api/types'
import { AutoDetectPanel } from '../components/AutoDetectPanel'
import { CanvasViewer, type Tool } from '../components/CanvasViewer'
import { DetectionList } from '../components/DetectionList'
import { ImageList } from '../components/ImageList'
import { LabelAssistPanel } from '../components/LabelAssistPanel'
import {
  AttributesPanel,
  ClassesPanel,
  HistoryPanel,
  LayersPanel,
  RawDataPanel,
  TagsPanel,
} from '../components/LeftPanels'
import { ShortcutHelpOverlay } from '../components/ShortcutHelpOverlay'
import { SmartSelectPanel } from '../components/SmartSelectPanel'
import { ToolPalette } from '../components/ToolPalette'
import { TopBar } from '../components/TopBar'
import { useHistory } from '../components/useHistory'
import { useKeyboardShortcuts, type Shortcut } from '../components/useKeyboardShortcuts'

interface ReviewPageProps {
  /** Active project (UI preset). null while projects are loading. */
  project: Project | null
  onBack: () => void
  onOpenExport: () => void
}

type LeftTab = 'images' | 'classes' | 'layers' | 'tags' | 'attributes' | 'history' | 'raw'

const LEFT_TABS: { key: LeftTab; label: string }[] = [
  { key: 'images', label: 'Images' },
  { key: 'classes', label: 'Classes' },
  { key: 'layers', label: 'Layers' },
  { key: 'tags', label: 'Tags' },
  { key: 'attributes', label: 'Attrs' },
  { key: 'history', label: 'History' },
  { key: 'raw', label: 'Raw' },
]

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
  const [trainOpen, setTrainOpen] = useState(false)
  const [leftTab, setLeftTab] = useState<LeftTab>('images')
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
  const cancelSmart = () => setSmartPoints([])

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
  const patchImage = usePatchImage(currentImageId ?? '')
  const createPolygon = useCreatePolygon(currentImageId ?? '')
  const copyAnnotations = useCopyAnnotations(currentImageId ?? '')
  const history = useHistory()

  // Stable mutate fns + recorder, so the action callbacks below stay stable.
  const patchMutate = patch.mutate
  const delMutate = del.mutate
  const createPolygonMutate = createPolygon.mutate
  const { record: recordHistory } = history

  // Review actions routed through here so each is recorded for undo/redo.
  const setApprovedFor = useCallback(
    (ann: Annotation, approved: boolean | null) => {
      const before = ann.approved
      patchMutate({ annotationId: ann.annotation_id, patch: { approved } })
      recordHistory({
        describe:
          approved === true
            ? 'Approve detection'
            : approved === false
              ? 'Reject detection'
              : 'Clear review',
        undo: () => patchMutate({ annotationId: ann.annotation_id, patch: { approved: before } }),
        redo: () => patchMutate({ annotationId: ann.annotation_id, patch: { approved } }),
      })
    },
    [patchMutate, recordHistory],
  )
  const setLabelFor = useCallback(
    (ann: Annotation, label: string) => {
      const before = ann.label
      patchMutate({ annotationId: ann.annotation_id, patch: { label } })
      recordHistory({
        describe: `Label → ${label}`,
        undo: () => patchMutate({ annotationId: ann.annotation_id, patch: { label: before } }),
        redo: () => patchMutate({ annotationId: ann.annotation_id, patch: { label } }),
      })
    },
    [patchMutate, recordHistory],
  )
  const deleteFor = useCallback(
    (ann: Annotation) => {
      delMutate({ annotationId: ann.annotation_id, imageId: ann.image_id })
      if (selectedAnnotationId === ann.annotation_id) setSelectedAnnotationId(null)
      // Undo recreates from the stored outline (best-effort: new id, mask re-
      // rasterised from the first polygon); redo deletes the recreated one.
      const recreated = { id: null as string | null }
      recordHistory({
        describe: 'Delete detection',
        undo: () => {
          const poly = ann.polygon[0]
          if (!poly) return
          const points: [number, number][] = []
          for (let i = 0; i + 1 < poly.length; i += 2) points.push([poly[i], poly[i + 1]])
          createPolygonMutate(
            { points, label: ann.label },
            {
              onSuccess: (created) => {
                recreated.id = created.annotation_id
                if (ann.approved !== null) {
                  patchMutate({
                    annotationId: created.annotation_id,
                    patch: { approved: ann.approved },
                  })
                }
              },
            },
          )
        },
        redo: () =>
          delMutate({ annotationId: recreated.id ?? ann.annotation_id, imageId: ann.image_id }),
      })
    },
    [delMutate, createPolygonMutate, patchMutate, recordHistory, selectedAnnotationId],
  )
  const repeatPrevious = () => {
    if (currentIndex > 0) copyAnnotations.mutate(imageList[currentIndex - 1].image_id)
  }
  // Record a creation (point/box/smart/polygon) so Ctrl+Z removes the new mask.
  // Undo deletes the created masks (lossless); redo re-draws them from their
  // outlines.
  const recordCreate = (created: Annotation[]) => {
    if (created.length === 0) return
    const imageId = created[0].image_id
    const ids = created.map((a) => a.annotation_id)
    recordHistory({
      describe: created.length === 1 ? 'Add detection' : `Add ${created.length} detections`,
      undo: () => ids.forEach((id) => delMutate({ annotationId: id, imageId })),
      redo: () => {
        created.forEach((a) => {
          const poly = a.polygon[0]
          if (!poly) return
          const points: [number, number][] = []
          for (let i = 0; i + 1 < poly.length; i += 2) points.push([poly[i], poly[i + 1]])
          createPolygonMutate(
            { points, label: a.label },
            {
              onSuccess: (c) => {
                if (a.approved !== null) {
                  patchMutate({ annotationId: c.annotation_id, patch: { approved: a.approved } })
                }
              },
            },
          )
        })
      },
    })
  }

  const finishSmart = () => {
    if (smartPoints.length === 0 || !currentImageId) return
    inferPoints.mutate(
      { points: smartPoints, simplify },
      {
        onSuccess: (anns) => {
          setSmartPoints([])
          recordCreate(anns)
        },
      },
    )
  }

  // Enter = finish, Esc = cancel while building a Smart Select (skip in inputs).
  useEffect(() => {
    if (currentTool !== 'smart' || smartPoints.length === 0) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (e.key === 'Enter') {
        e.preventDefault()
        finishSmart()
      } else if (e.key === 'Escape') {
        e.preventDefault()
        setSmartPoints([])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // finishSmart closes over the latest state; re-subscribe on the inputs it uses.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTool, smartPoints, simplify, currentImageId])

  // Image navigator (also used by the Shift+N/P shortcuts).
  const imageList = images ?? []
  const currentIndex = imageList.findIndex((i) => i.image_id === currentImageId)
  const goToImage = (delta: number) => {
    if (imageList.length === 0 || currentIndex < 0) return
    const next = (currentIndex + delta + imageList.length) % imageList.length
    setCurrentImageId(imageList[next].image_id)
  }
  const toggleDone = () => {
    if (currentImage) patchImage.mutate({ done: !currentImage.done })
  }

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
      if (selectedAnn) setApprovedFor(selectedAnn, approved)
    }
    const deleteSelected = () => {
      if (selectedAnn) deleteFor(selectedAnn)
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
        combo: 'Shift+G',
        label: 'Polygon tool',
        group: 'Tools',
        handler: () => setCurrentTool('polygon'),
      },
      {
        combo: 'Shift+E',
        label: 'Open export dialog',
        group: 'General',
        handler: onOpenExport,
      },
      {
        combo: 'Ctrl+Z',
        label: 'Undo',
        group: 'General',
        handler: history.undo,
      },
      {
        combo: 'Ctrl+Y',
        label: 'Redo',
        group: 'General',
        handler: history.redo,
      },
    ]
  }, [
    images,
    annotations,
    currentImageId,
    selectedAnnotationId,
    onOpenExport,
    setApprovedFor,
    deleteFor,
    history.undo,
    history.redo,
  ])

  // Disable global shortcuts while the help overlay is open — it owns its own
  // dismiss listener so any key closes it instead of triggering a page action.
  useKeyboardShortcuts(shortcuts, !helpOpen)

  return (
    // grid with sensible minimums: at narrow widths the page gets a horizontal
    // scrollbar instead of crushing the canvas column.
    <div className="grid h-full min-h-0 min-w-[900px] grid-cols-[240px_minmax(420px,1fr)_320px] overflow-auto">
      {/* Left: view-switcher (images / classes / layers / tags / raw) */}
      <aside className="flex min-h-0 min-w-0 flex-col border-r border-slate-200 bg-white">
        <div className="flex-shrink-0 border-b border-slate-200">
          <div className="flex items-center justify-between px-3 pt-1.5">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Panel
            </span>
            <button
              type="button"
              onClick={onBack}
              className="text-xs text-slate-500 hover:text-slate-800"
            >
              ← upload
            </button>
          </div>
          <div className="flex flex-wrap gap-0.5 px-1 pb-1">
            {LEFT_TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                onClick={() => setLeftTab(t.key)}
                className={`flex-1 rounded px-1 py-0.5 text-xs transition ${
                  leftTab === t.key
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          {leftTab === 'images' && (
            <ImageList selectedId={currentImageId} onSelect={setCurrentImageId} />
          )}
          {leftTab === 'classes' && (
            <ClassesPanel annotations={annotations ?? []} projectClasses={classes} />
          )}
          {leftTab === 'layers' && (
            <LayersPanel
              annotations={annotations ?? []}
              selectedId={selectedAnnotationId}
              onSelect={setSelectedAnnotationId}
            />
          )}
          {leftTab === 'tags' &&
            (currentImage ? (
              <TagsPanel
                tags={currentImage.tags}
                onChange={(tags) => patchImage.mutate({ tags })}
              />
            ) : (
              <p className="p-3 text-sm text-slate-500">Select an image first.</p>
            ))}
          {leftTab === 'attributes' && (
            <AttributesPanel
              annotation={
                (annotations ?? []).find((a) => a.annotation_id === selectedAnnotationId) ?? null
              }
            />
          )}
          {leftTab === 'history' && <HistoryPanel past={history.past} future={history.future} />}
          {leftTab === 'raw' && <RawDataPanel annotations={annotations ?? []} />}
        </div>
      </aside>

      {/* Centre: canvas + tool palette */}
      <section className="relative flex min-h-0 min-w-0 flex-col bg-slate-100">
        <div className="flex flex-shrink-0 items-center gap-3 border-b border-slate-200 bg-white px-4 py-2">
          <TopBar
            projectLabel={project?.label ?? null}
            fileName={currentImage?.file_name ?? null}
            index={currentIndex}
            total={imageList.length}
            done={currentImage?.done ?? false}
            onPrev={() => goToImage(-1)}
            onNext={() => goToImage(1)}
            onToggleDone={toggleDone}
            onTrain={() => setTrainOpen(true)}
          />
          <ToolPalette current={currentTool} onChange={setCurrentTool} />
          <div className="flex flex-shrink-0 items-center gap-0.5">
            <button
              type="button"
              onClick={history.undo}
              disabled={!history.canUndo}
              title="Undo (Ctrl+Z)"
              className="grid h-7 w-7 place-items-center rounded border border-slate-200 text-slate-600 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
            >
              ↶
            </button>
            <button
              type="button"
              onClick={history.redo}
              disabled={!history.canRedo}
              title="Redo (Ctrl+Y)"
              className="grid h-7 w-7 place-items-center rounded border border-slate-200 text-slate-600 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
            >
              ↷
            </button>
          </div>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            title="Shortcuts (Shift+?)"
            className="flex-shrink-0 rounded border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-100"
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
              onAnnotationsCreated={recordCreate}
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
            <button
              type="button"
              onClick={repeatPrevious}
              disabled={currentIndex <= 0 || copyAnnotations.isPending}
              title="Copy the previous image's annotations onto this one"
              className="mt-2 w-full rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
            >
              ⧉ Repeat previous image
            </button>
            <div className="mt-2">
              <LabelAssistPanel />
            </div>
          </div>
        )}
        <div className="min-h-0 flex-1">
          {currentImage && (
            <DetectionList
              imageId={currentImage.image_id}
              classes={classes}
              selectedAnnotationId={selectedAnnotationId}
              onSelectAnnotation={setSelectedAnnotationId}
              onSetApproved={setApprovedFor}
              onSetLabel={setLabelFor}
              onDelete={deleteFor}
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

      {trainOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
          onClick={() => setTrainOpen(false)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-[26rem] rounded-lg bg-white p-6 shadow-xl"
          >
            <h2 className="text-lg font-semibold text-slate-900">Train a model</h2>
            <p className="mt-2 text-sm text-slate-600">
              Custom model training runs on your approved annotations and lands in the next sprint.
              For now, keep labeling and approving — those become the training set.
            </p>
            <div className="mt-5 flex justify-end">
              <button
                type="button"
                onClick={() => setTrainOpen(false)}
                className="rounded bg-slate-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-slate-700"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
