interface SmartSelectPanelProps {
  pointCount: number
  simplify: number
  onSimplifyChange: (value: number) => void
  onFinish: () => void
  onCancel: () => void
  busy: boolean
}

/**
 * Right-sidebar controls for Smart Select (SAM multi-point).
 *
 * The canvas owns the interaction (hover preview, click to add include points,
 * Shift-click for exclude points). This panel surfaces the running point count,
 * the Simplify slider (polygon smoothing), and the Finish / Cancel actions —
 * mirrored by Enter / Esc on the keyboard.
 */
export function SmartSelectPanel({
  pointCount,
  simplify,
  onSimplifyChange,
  onFinish,
  onCancel,
  busy,
}: SmartSelectPanelProps) {
  return (
    <div className="flex flex-col gap-2 rounded border border-sky-200 bg-sky-50 p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-semibold text-sky-900">Smart Select</span>
        <span className="text-xs text-sky-700">
          {pointCount} point{pointCount === 1 ? '' : 's'}
        </span>
      </div>
      <p className="text-xs text-slate-600">
        Hover to preview · click to add · Shift-click to exclude · Enter to finish · Esc to cancel.
      </p>
      <label className="flex items-center gap-2 text-xs text-slate-600">
        <span className="w-16 flex-shrink-0">simplify</span>
        <input
          type="range"
          min={0}
          max={10}
          step={0.5}
          value={simplify}
          onChange={(e) => onSimplifyChange(Number(e.target.value))}
          className="flex-1"
        />
        <span className="w-8 text-right font-mono">{simplify.toFixed(1)}</span>
      </label>
      <div className="flex gap-1">
        <button
          type="button"
          onClick={onFinish}
          disabled={pointCount === 0 || busy}
          className="flex-1 rounded bg-sky-600 px-2 py-1 text-xs font-medium text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {busy ? 'Saving…' : 'Finish (↵)'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={pointCount === 0 || busy}
          className="flex-1 rounded bg-slate-200 px-2 py-1 text-xs font-medium text-slate-700 transition hover:bg-slate-300 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Cancel (Esc)
        </button>
      </div>
    </div>
  )
}
