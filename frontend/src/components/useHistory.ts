import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * One reversible action. `undo` / `redo` perform the actual API mutations; the
 * caller builds these closures when recording so the hook stays API-agnostic.
 */
export interface HistoryAction {
  describe: string
  undo: () => void
  redo: () => void
}

export interface History {
  record: (action: HistoryAction) => void
  undo: () => void
  redo: () => void
  clear: () => void
  canUndo: boolean
  canRedo: boolean
  past: HistoryAction[]
  future: HistoryAction[]
}

/**
 * A small undo/redo stack for review actions (approve/reject, relabel, delete).
 *
 * The stacks live in state so render-time values (`canUndo`, the History panel)
 * stay correct and lint-clean; mirror refs give the stable `undo`/`redo`/`record`
 * callbacks synchronous access to the latest stacks without stale closures.
 */
export function useHistory(): History {
  const [past, setPast] = useState<HistoryAction[]>([])
  const [future, setFuture] = useState<HistoryAction[]>([])
  const pastRef = useRef(past)
  const futureRef = useRef(future)
  useEffect(() => {
    pastRef.current = past
  }, [past])
  useEffect(() => {
    futureRef.current = future
  }, [future])

  const record = useCallback((action: HistoryAction) => {
    setPast((p) => [...p, action])
    setFuture([]) // a fresh action invalidates the redo branch
  }, [])

  const undo = useCallback(() => {
    const stack = pastRef.current
    if (stack.length === 0) return
    const action = stack[stack.length - 1]
    setPast(stack.slice(0, -1))
    setFuture([action, ...futureRef.current])
    action.undo()
  }, [])

  const redo = useCallback(() => {
    const stack = futureRef.current
    if (stack.length === 0) return
    const action = stack[0]
    setFuture(stack.slice(1))
    setPast([...pastRef.current, action])
    action.redo()
  }, [])

  const clear = useCallback(() => {
    setPast([])
    setFuture([])
  }, [])

  return {
    record,
    undo,
    redo,
    clear,
    canUndo: past.length > 0,
    canRedo: future.length > 0,
    past,
    future,
  }
}
