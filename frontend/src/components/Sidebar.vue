<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Document, Search } from '@element-plus/icons-vue'

const props = defineProps({
  headerOffset: {
    type: Number,
    default: 200,
  },
})

const route = useRoute()
const router = useRouter()

const activePath = computed(() => {
  if (route.path.startsWith('/docs')) return '/docs'
  if (route.path.startsWith('/history')) return '/history'
  if (route.path.startsWith('/search-advanced')) return '/search-advanced'
  return '/search'
})

const handleSelect = (index) => {
  if (index && index !== route.path) {
    router.push(index)
  }
}

const sidebarStyle = computed(() => ({
  top: `${props.headerOffset}px`,
  height: `calc(100vh - ${props.headerOffset}px)`,
}))
</script>

<template>
  <nav class="sidebar" :style="sidebarStyle">
    <el-menu
      :default-active="activePath"
      class="menu"
      router
      @select="handleSelect"
    >
      <el-menu-item index="/docs">
        <el-icon><Document /></el-icon>
        <span>Docs</span>
      </el-menu-item>
      <el-menu-item index="/search">
        <el-icon><Search /></el-icon>
        <span>Standard Search</span>
      </el-menu-item>
      <el-menu-item index="/search-advanced">
        <el-icon><Search /></el-icon>
        <span>Advanced Search</span>
      </el-menu-item>
      <el-menu-item index="/history">
        <el-icon><Document /></el-icon>
        <span>History</span>
      </el-menu-item>
    </el-menu>
  </nav>
</template>

<style scoped>
.sidebar {
  position: fixed;
  left: 0;
  width: 250px;
  background: rgb(175, 206, 235);
  border-right: 1px solid #e6e6e6;
  box-sizing: border-box;
  z-index: 999;
  padding-top: 12px;
  transition: top 0.28s ease, height 0.28s ease;
}

.menu {
  border-right: none;
  background-color: transparent;
}

:deep(.el-menu-item) {
  height: 48px;
  line-height: 48px;
}

:deep(.el-menu-item.is-active) {
  background-color: #e8f4f8 !important;
  color: #1989fa !important;
  border-left: 3px solid #1989fa;
}

:deep(.el-menu-item:hover) {
  background-color: #e6f7ff !important;
}
</style>
