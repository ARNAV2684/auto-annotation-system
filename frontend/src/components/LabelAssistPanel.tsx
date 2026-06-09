/**
 * Label Assist (model-assisted pre-labeling) — a stub until Sprint 7 wires up
 * real training + inference. Collapsed by default so it stays out of the way.
 */
export function LabelAssistPanel() {
  return (
    <details className="rounded border border-slate-200 bg-white text-sm">
      <summary className="cursor-pointer px-3 py-2 font-medium text-slate-700">
        Label Assist
      </summary>
      <div className="space-y-3 border-t border-slate-200 p-3 text-xs text-slate-500">
        <div>
          <div className="font-semibold text-slate-600">Your Models</div>
          <p>
            No trained models yet. Annotate, approve, and train (Sprint 7) — your approved labels
            become the training set, then the model pre-labels new images.
          </p>
        </div>
        <div>
          <div className="font-semibold text-slate-600">Public Models</div>
          <p>Coming soon — run a public detector (e.g. MS COCO) to pre-label common objects.</p>
        </div>
      </div>
    </details>
  )
}
