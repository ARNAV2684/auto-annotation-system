import { useState } from 'react'
import { useDeleteSession, useImages } from './api/hooks'
import { ConfirmDialog } from './components/ConfirmDialog'
import { ExportDialog } from './components/ExportDialog'
import { Header } from './components/Header'
import { ReviewPage } from './pages/ReviewPage'
import { UploadPage } from './pages/UploadPage'

// Hardcoded for v1; mirrors config.yaml#classes on the backend. A future
// GET /api/classes route would replace this.
const CLASSES = ['tree', 'palm_tree', 'shrub']

/**
 * Top-level shell. Owns the navigation (upload <-> review) and modal state
 * (Export dialog, Clear Session confirm).
 */
export default function App() {
  const { data: images } = useImages()
  const deleteSession = useDeleteSession()
  const [view, setView] = useState<'upload' | 'review'>('upload')
  const [exportOpen, setExportOpen] = useState(false)
  const [clearOpen, setClearOpen] = useState(false)

  return (
    <div className="flex h-full flex-col">
      <Header onClearSession={() => setClearOpen(true)} />
      <main className="flex-1 overflow-hidden">
        {view === 'review' && (images?.length ?? 0) > 0 ? (
          <ReviewPage
            classes={CLASSES}
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
