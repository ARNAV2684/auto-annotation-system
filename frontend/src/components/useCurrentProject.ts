import { useState } from 'react'
import { useProjects } from '../api/hooks'
import type { Project } from '../api/types'

const STORAGE_KEY = 'autoannot:currentProjectId'

/**
 * Hook that returns the currently active project + a setter that persists
 * the choice to localStorage. Defaults to the first project if no choice was
 * saved or the saved choice no longer exists.
 *
 * Lives in its own file so the ProjectSelector component file only exports
 * components (required by react-refresh / HMR).
 */
export function useCurrentProject(): {
  current: Project | null
  setCurrentId: (id: string) => void
  projects: Project[] | undefined
} {
  const { data: projects } = useProjects()
  const [currentId, setCurrentIdState] = useState<string | null>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY)
    } catch {
      return null
    }
  })

  const setCurrentId = (id: string) => {
    setCurrentIdState(id)
    try {
      localStorage.setItem(STORAGE_KEY, id)
    } catch {
      // localStorage may be disabled — degrade silently to session-only.
    }
  }

  let current: Project | null = null
  if (projects && projects.length > 0) {
    current = projects.find((p) => p.id === currentId) ?? projects[0]
  }

  return { current, setCurrentId, projects }
}
