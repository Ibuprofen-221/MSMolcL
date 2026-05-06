import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 6006,
    proxy: {
      '/api-backend': {
        target: 'http://127.0.0.1:6008',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api-backend/, '')
      }
    }
  },
  preview: {
    host: '127.0.0.1',
    port: 6008,
  },
})
