<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Document, Search } from '@element-plus/icons-vue'
import { clearAuth, clearBusinessSession, getCurrentUser } from '../utils/storage'

const route = useRoute()
const router = useRouter()

const menus = [
  { path: '/docs', label: 'Docs', icon: Document },
  { path: '/search', label: 'Standard Search', icon: Search },
  { path: '/search-advanced', label: 'Advanced Search', icon: Search },
  { path: '/history', label: 'History', icon: Document },
]

const activePath = computed(() => {
  if (route.path.startsWith('/docs')) return '/docs'
  if (route.path.startsWith('/history')) return '/history'
  if (route.path.startsWith('/search-advanced')) return '/search-advanced'
  return '/search'
})

const username = computed(() => getCurrentUser()?.username || 'Unknown')

const go = (path) => {
  if (path !== route.path) {
    router.push(path)
  }
}

const logout = () => {
  clearBusinessSession()
  clearAuth()
  router.replace('/login')
}
</script>

<template>
  <header class="top-nav">
    <div class="brand">MSMoLCl</div>
    <div class="menu-grid">
      <button
        v-for="(item, index) in menus"
        :key="item.path"
        type="button"
        class="menu-item"
        :class="{ active: activePath === item.path, divider: index > 0 }"
        @click="go(item.path)"
      >
        <el-icon><component :is="item.icon" /></el-icon>
        <span>{{ item.label }}</span>
      </button>
    </div>
    <div class="auth-box">
      <span class="user">{{ username }}</span>
      <button type="button" class="logout-btn" @click="logout">Logout</button>
    </div>
  </header>
</template>

<style scoped>
.top-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 100px;
  background: #3e7ab6;
  display: flex;
  align-items: center;
  z-index: 1100;
}

.brand {
  width: 220px;
  padding-left: 16px;
  font-size: 34px;
  font-weight: 800;
  color: #000000;
  user-select: none;
}

.menu-grid {
  flex: 1;
  height: 100%;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.menu-item {
  border: none;
  border-radius: 0;
  background: transparent;
  color: #000000;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-size: 17px;
  cursor: pointer;
  transition: background-color 0.2s ease, color 0.2s ease;
}

.menu-item.divider {
  position: relative;
}

.menu-item.divider::before {
  content: '';
  position: absolute;
  left: 0;
  top: 36%;
  height: 28%;
  width: 1px;
  background: rgba(0, 0, 0, 0.6);
}

.menu-item:hover {
  color: #ffffff;
}

.menu-item.active {
  color: #ffffff;
  background: #5287aa;
}

.auth-box {
  width: 210px;
  padding-right: 14px;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
}

.user {
  color: #0f172a;
  font-size: 14px;
  font-weight: 600;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.logout-btn {
  border: 1px solid rgba(0, 0, 0, 0.45);
  background: transparent;
  color: #000;
  border-radius: 4px;
  height: 32px;
  padding: 0 12px;
  cursor: pointer;
}

.logout-btn:hover {
  color: #fff;
  border-color: #fff;
}

:deep(.el-icon) {
  font-size: 15px;
}
</style>
