import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 6006,
    
    allowedHosts: [
      'u742891-9109-afb63769.bjb2.seetacloud.com',
      'u742891-8b98-66121702.bjb2.seetacloud.com' // 顺便把你之前那个域名也加上，防止以后变动
    ],
    
    proxy: {
      // 告诉 Vite 拦截所有以 /api-backend 开头的请求
      '/api-backend': {
        target: 'http://127.0.0.1:6008', // 真实的后端地址
        changeOrigin: true, // 允许改变 Origin 头部，欺骗后端这就像是一个直接发往后端的请求
        rewrite: (path) => path.replace(/^\/api-backend/, '') // 转发前去掉这个前缀，因为后端路由里没有 /api-backend
      }
    }
  },
  preview: {
    host: '127.0.0.1',
    port: 6008,
  },
})
