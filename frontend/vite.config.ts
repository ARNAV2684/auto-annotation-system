import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Forward every /api/* request from the Vite dev server (5173) to the
    // FastAPI backend (8000). Lets frontend code use bare paths like '/api/upload'
    // without hardcoding the host - the same path also works in Docker behind nginx.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
