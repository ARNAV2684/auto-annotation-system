import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'

// One QueryClient for the whole app. Wraps every cache, retry, and refetch policy.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000, // re-fetch after 30s of inactivity
      retry: 1, // single retry on transient failures
      refetchOnWindowFocus: false,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
