interface SmartSelectPanelProps {
  objectCount: number
  onFinish: () => void
  onCancel: () => void
  busy: boolean
}

/**
 * Right-sidebar controls for Smart Select (SAM collect-many).
 *
 * The canvas owns the interaction: hover previews the object under the cursor,
 * a click collects it (and hovering resumes for the next object). This panel
 * shows how many are collected and the Finish / Cancel actions — mirrored by
 * Enter / Esc on the keyboard. Finish sends them all to the detections list.
 */
export function SmartSelectPanel({ objectCount, onFinish, onCancel, busy }: SmartSelectPanelProps) {
  return (
    <div className="flex flex-col gap-2 rounded border border-sky-200 bg-sky-50 p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-semibold text-sky-900">Smart Select</span>
        <span className="text-xs text-sky-700">
          {objectCount} object{objectCount === 1 ? '' : 's'} collected
        </span>
      </div>
      <p className="text-xs text-slate-600">
        Hover to preview an object · click to collect it · keep going · Enter to send all to
        Detections · Esc to clear.
      </p>
      <div className="flex gap-1">
        <button
          type="button"
          onClick={onFinish}
          disabled={objectCount === 0 || busy}
          className="flex-1 rounded bg-sky-600 px-2 py-1 text-xs font-medium text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {busy ? 'Saving…' : `Finish (↵) · ${objectCount}`}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={objectCount === 0 || busy}
          className="flex-1 rounded bg-slate-200 px-2 py-1 text-xs font-medium text-slate-700 transition hover:bg-slate-300 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Cancel (Esc)
        </button>
      </div>
    </div>
  )
}
