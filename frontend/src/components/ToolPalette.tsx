import type { Tool } from './CanvasViewer'

interface ToolPaletteProps {
  current: Tool
  onChange: (tool: Tool) => void
}

/** Compact toolbar to switch between point-prompt and box-prompt modes. */
export function ToolPalette({ current, onChange }: ToolPaletteProps) {
  return (
    <div className="inline-flex items-center gap-1 rounded border border-slate-200 bg-white p-1 shadow-sm">
      <ToolButton
        label="Smart"
        hint="hover to preview, click to add points, Enter to finish"
        active={current === 'smart'}
        onClick={() => onChange('smart')}
      />
      <ToolButton
        label="Point"
        hint="click to segment"
        active={current === 'point'}
        onClick={() => onChange('point')}
      />
      <ToolButton
        label="Box"
        hint="drag a rectangle"
        active={current === 'box'}
        onClick={() => onChange('box')}
      />
      <ToolButton
        label="Polygon"
        hint="click vertices, Enter or double-click to finish"
        active={current === 'polygon'}
        onClick={() => onChange('polygon')}
      />
    </div>
  )
}

function ToolButton({
  label,
  hint,
  active,
  onClick,
}: {
  label: string
  hint: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={hint}
      className={`rounded px-3 py-1 text-sm transition ${
        active ? 'bg-sky-600 text-white' : 'text-slate-600 hover:bg-slate-100'
      }`}
    >
      {label}
    </button>
  )
}
