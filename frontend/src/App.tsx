import { useState } from 'react'
import { useDeleteSession, useImages } from './api/hooks'
import { Header } from './components/Header'
import { ReviewPage } from './pages/ReviewPage'
import { UploadPage } from './pages/UploadPage'

// Hardcoded for v1; mirrors config.yaml#classes on the backend. A future
// GET /api/classes route would replace this.
const CLASSES = ['tree', 'palm_tree', 'shrub']

/**
 * Top-level shell. Owns the "are we uploading or reviewing?" navigation.
 *
 * Upload page → Review page once the user clicks "Start Review".
 * Review page → Upload page via its own back button.
 */
export default function App() {
  const { data: images } = useImages()
  const deleteSession = useDeleteSession()
  const [view, setView] = useState<'upload' | 'review'>('upload')

  return (
    <div className="flex h-full flex-col">
      <Header onClearSession={() => deleteSession.mutate()} />
      <main className="flex-1 overflow-hidden">
        {view === 'review' && (images?.length ?? 0) > 0 ? (
          <ReviewPage classes={CLASSES} onBack={() => setView('upload')} />
        ) : (
          <UploadPage imageCount={images?.length ?? 0} onStartReview={() => setView('review')} />
        )}
      </main>
    </div>
  )
}
