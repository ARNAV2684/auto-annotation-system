import { useState } from 'react'
import { useDeleteSession, useImages } from './api/hooks'
import { ConfirmDialog } from './components/ConfirmDialog'
import { ExportDialog } from './components/ExportDialog'
import { Header } from './components/Header'
import { useCurrentProject } from './components/useCurrentProject'
import { ReviewPage } from './pages/ReviewPage'
import { UploadPage } from './pages/UploadPage'

/**
 * Top-level shell. Owns:
 *   - upload <-> review view navigation
 *   - modal state (Export dialog, Clear Session confirm)
 *   - currently active Project (persisted to localStorage by useCurrentProject)
 */
export default function App() {
  const { data: images } = useImages()
  const deleteSession = useDeleteSession()
  const { current: currentProject, setCurrentId: setProjectId } = useCurrentProject()
  const [view, setView] = useState<'upload' | 'review'>('upload')
  const [exportOpen, setExportOpen] = useState(false)
  const [clearOpen, setClearOpen] = useState(false)

  return (
    <div className="flex h-screen min-h-0 w-full min-w-0 flex-col overflow-hidden">
      <Header
        currentProject={currentProject}
        onProjectChange={setProjectId}
        onClearSession={() => setClearOpen(true)}
      />
      <main className="min-h-0 min-w-0 flex-1 overflow-hidden">
        {view === 'review' && (images?.length ?? 0) > 0 ? (
          <ReviewPage
            project={currentProject}
            onBack={() => setView('upload')}
            onOpenExport={() => setExportOpen(true)}
          />
        ) : (
          <UploadPage imageCount={images?.length ?? 0} onStartReview={() => setView('review')} />
        )}
      </main>

      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} />
      <ConfirmDialog
        open={clearOpen}
        title="Clear session?"
        message="This deletes every uploaded image, mask, and annotation on the server. It cannot be undone."
        confirmLabel="Clear everything"
        confirmDanger
        onConfirm={() => {
          deleteSession.mutate()
          setView('upload')
        }}
        onClose={() => setClearOpen(false)}
      />
    </div>
  )
}
