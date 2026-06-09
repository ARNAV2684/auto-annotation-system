interface TopBarProps {
  projectLabel: string | null
  fileName: string | null
  /** 0-based index of the current image, or -1 if none. */
  index: number
  total: number
  done: boolean
  onPrev: () => void
  onNext: () => void
  onToggleDone: () => void
  onTrain: () => void
}

/**
 * Workspace top bar: project breadcrumb · filename · image navigator (◀ n/N ▶) ·
 * a "done / reviewed" toggle, and the TRAIN button (a stub until Sprint 7 wires
 * up real model training).
 */
export function TopBar({
  projectLabel,
  fileName,
  index,
  total,
  done,
  onPrev,
  onNext,
  onToggleDone,
  onTrain,
}: TopBarProps) {
  const hasImage = index >= 0 && total > 0
  return (
    <div className="flex min-w-0 flex-1 items-center gap-3">
      {projectLabel && (
        <span className="flex-shrink-0 text-xs font-medium text-slate-400">{projectLabel} /</span>
      )}
      <span className="min-w-0 truncate text-sm font-medium text-slate-700">
        {fileName ?? 'No image selected'}
      </span>

      <div className="ml-auto flex flex-shrink-0 items-center gap-2">
        <div className="flex items-center gap-1 rounded border border-slate-200 bg-white px-1">
          <NavButton
            label="◀"
            title="Previous image (Shift+P)"
            onClick={onPrev}
            disabled={!hasImage}
          />
          <span className="min-w-14 text-center font-mono text-xs text-slate-600">
            {hasImage ? `${index + 1} / ${total}` : '0 / 0'}
          </span>
          <NavButton label="▶" title="Next image (Shift+N)" onClick={onNext} disabled={!hasImage} />
        </div>

        <button
          type="button"
          onClick={onToggleDone}
          disabled={!hasImage}
          title="Mark this image reviewed"
          className={`flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
            done
              ? 'border-emerald-300 bg-emerald-500 text-white'
              : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-100'
          }`}
        >
          ✓ {done ? 'Done' : 'Mark done'}
        </button>

        <button
          type="button"
          onClick={onTrain}
          title="Train a model from your labels"
          className="rounded bg-indigo-600 px-3 py-1 text-xs font-semibold text-white shadow transition hover:bg-indigo-500"
        >
          TRAIN
        </button>
      </div>
    </div>
  )
}

function NavButton({
  label,
  title,
  onClick,
  disabled,
}: {
  label: string
  title: string
  onClick: () => void
  disabled: boolean
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className="grid h-6 w-6 place-items-center rounded text-slate-600 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {label}
    </button>
  )
}
