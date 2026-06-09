import { useState } from 'react'
import { useExport, useProjects, useStats } from '../api/hooks'
import type { ExportFormat } from '../api/types'
import { useCurrentProject } from './useCurrentProject'

interface ExportDialogProps {
  open: boolean
  onClose: () => void
}

const FORMAT_OPTIONS: { value: Exclude<ExportFormat, 'all'>; label: string; hint: string }[] = [
  { value: 'yolo', label: 'YOLO', hint: '.txt per image, normalised polygons' },
  { value: 'coco', label: 'COCO', hint: 'one annotations.json (polygon + RLE)' },
  { value: 'voc', label: 'Pascal VOC', hint: '.xml per image, bounding boxes' },
]

/**
 * Modal: pick a project to scope the class list, pick formats, choose
 * only-approved vs everything, hit Download.
 *
 * On success we fabricate an `<a href download>` and click it, so the browser
 * saves the zip to the user's Downloads folder. If the export skipped any
 * annotations (their label wasn't in the chosen project's class list), we keep
 * the dialog open and show a warning instead of silently closing.
 */
export function ExportDialog({ open, onClose }: ExportDialogProps) {
  const [yolo, setYolo] = useState(true)
  const [coco, setCoco] = useState(true)
  const [voc, setVoc] = useState(true)
  const [onlyApproved, setOnlyApproved] = useState(true)
  const [warning, setWarning] = useState<{ count: number; labels: string[] } | null>(null)
  const exportRun = useExport()
  const { data: stats } = useStats()
  const { data: projects } = useProjects()
  const { current } = useCurrentProject()

  // Seed the project picker to the active project once (React 19 "derive state
  // during render" — no effect). '' means "all projects (union)".
  const [projectId, setProjectId] = useState('')
  const [seeded, setSeeded] = useState(false)
  if (!seeded && current) {
    setSeeded(true)
    setProjectId(current.id)
  }

  if (!open) return null

  const handleClose = () => {
    setWarning(null)
    onClose()
  }

  const selected: Exclude<ExportFormat, 'all'>[] = []
  if (yolo) selected.push('yolo')
  if (coco) selected.push('coco')
  if (voc) selected.push('voc')

  const canSubmit = selected.length > 0 && (stats?.num_annotations ?? 0) > 0

  const handleDownload = async () => {
    setWarning(null)
    const result = await exportRun.mutateAsync({
      formats: selected,
      only_approved: onlyApproved,
      project_id: projectId || null,
    })
    // Trigger the browser download.
    const a = document.createElement('a')
    a.href = result.download_url
    a.download = `${result.run_id}.zip`
    document.body.appendChild(a)
    a.click()
    a.remove()
    if (result.num_dropped > 0) {
      setWarning({ count: result.num_dropped, labels: result.dropped_labels })
    } else {
      handleClose()
    }
  }

  const annotationCount = onlyApproved ? (stats?.num_approved ?? 0) : (stats?.num_annotations ?? 0)

  return (
    <Backdrop onClose={handleClose}>
      <div className="w-[28rem] rounded-lg bg-white p-6 shadow-xl">
        <h2 className="text-lg font-semibold text-slate-900">Export annotations</h2>
        <p className="mt-1 text-sm text-slate-500">
          Builds a zip with the chosen formats and the included annotations.
        </p>

        <div className="mt-5">
          <label className="block text-sm font-medium text-slate-700">Project</label>
          <select
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
            className="mt-1 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-800"
          >
            <option value="">All projects (every class list)</option>
            {(projects ?? []).map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          <p className="mt-1 text-xs text-slate-500">
            Only annotations whose label is in the chosen project&apos;s class list are exported.
          </p>
        </div>

        <div className="mt-5 space-y-2">
          <FormatRow checked={yolo} onChange={setYolo} option={FORMAT_OPTIONS[0]} />
          <FormatRow checked={coco} onChange={setCoco} option={FORMAT_OPTIONS[1]} />
          <FormatRow checked={voc} onChange={setVoc} option={FORMAT_OPTIONS[2]} />
        </div>

        <label className="mt-5 flex items-center gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            checked={onlyApproved}
            onChange={(e) => setOnlyApproved(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500"
          />
          Only approved annotations
        </label>

        <div className="mt-3 rounded bg-slate-50 px-3 py-2 text-sm">
          <span className="text-slate-600">Up to</span>{' '}
          <span className="font-semibold text-slate-900">
            {annotationCount} annotation{annotationCount === 1 ? '' : 's'}
          </span>{' '}
          <span className="text-slate-600">
            across {stats?.num_images ?? 0} image{stats?.num_images === 1 ? '' : 's'}
          </span>
        </div>

        {warning && (
          <div className="mt-3 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            <span className="font-semibold">{warning.count}</span> annotation
            {warning.count === 1 ? '' : 's'} skipped — label
            {warning.labels.length === 1 ? '' : 's'}{' '}
            {warning.labels.map((label, i) => (
              <span key={label}>
                <code className="rounded bg-amber-100 px-1">{label}</code>
                {i < warning.labels.length - 1 ? ', ' : ''}
              </span>
            ))}{' '}
            not in this project&apos;s class list. Pick a different project or relabel them, then
            export again.
          </div>
        )}

        {exportRun.isError && (
          <div className="mt-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {exportRun.error?.message}
          </div>
        )}

        <div className="mt-6 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={handleClose}
            disabled={exportRun.isPending}
            className="rounded px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
          >
            {warning ? 'Close' : 'Cancel'}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            disabled={!canSubmit || exportRun.isPending}
            className="rounded bg-sky-600 px-4 py-1.5 text-sm font-medium text-white shadow transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {exportRun.isPending ? 'Building zip…' : 'Download zip'}
          </button>
        </div>
      </div>
    </Backdrop>
  )
}

function FormatRow({
  checked,
  onChange,
  option,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  option: { value: string; label: string; hint: string }
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2 rounded border border-slate-200 px-3 py-2 transition hover:border-sky-300">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-500"
      />
      <div className="min-w-0">
        <div className="text-sm font-medium text-slate-800">{option.label}</div>
        <div className="text-xs text-slate-500">{option.hint}</div>
      </div>
    </label>
  )
}

function Backdrop({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onClick={onClose}
    >
      <div onClick={(e) => e.stopPropagation()}>{children}</div>
    </div>
  )
}
