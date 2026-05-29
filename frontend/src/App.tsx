import { useState } from 'react'
import { useDeleteSession, useImages } from './api/hooks'
import { Header } from './components/Header'
import { UploadPage } from './pages/UploadPage'

/**
 * Top-level shell.
 *
 * PR A (scaffold): shows the upload page. If any images already exist in the
 * session, shows a placeholder until PR B (review UI) lands.
 */
export default function App() {
  const { data: images } = useImages()
  const deleteSession = useDeleteSession()
  const [showReviewPlaceholder, setShowReviewPlaceholder] = useState(false)

  const hasImages = (images?.length ?? 0) > 0
  const inReview = hasImages && showReviewPlaceholder

  return (
    <div className="flex h-full flex-col">
      <Header onClearSession={() => deleteSession.mutate()} />
      <main className="flex-1 overflow-auto">
        {inReview ? (
          <ReviewPlaceholder onBack={() => setShowReviewPlaceholder(false)} />
        ) : (
          <UploadPage
            imageCount={images?.length ?? 0}
            onStartReview={() => setShowReviewPlaceholder(true)}
          />
        )}
      </main>
    </div>
  )
}

/** Filler shown after upload until PR B replaces it with the real ReviewPage. */
function ReviewPlaceholder({ onBack }: { onBack: () => void }) {
  return (
    <div className="mx-auto mt-16 max-w-xl rounded border border-slate-200 bg-white p-8 text-center shadow-sm">
      <h2 className="mb-2 text-xl font-semibold text-slate-900">Review UI coming in PR B</h2>
      <p className="text-slate-600">
        The scaffold (PR A) handles uploads and live stats. Click below to go back to the upload
        page; the canvas + masks + approve buttons land in PR B.
      </p>
      <button
        type="button"
        onClick={onBack}
        className="mt-6 rounded bg-slate-900 px-4 py-2 text-sm text-white transition hover:bg-slate-700"
      >
        ← Back to upload
      </button>
    </div>
  )
}
