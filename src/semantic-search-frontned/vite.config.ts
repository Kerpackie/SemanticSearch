import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5134',
        changeOrigin: true,
        secure: false,
        timeout: 300000, // 5 minutes for long-running ML model operations
      }
    }
  }
})
