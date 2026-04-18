<script setup>
import { computed, provide, ref } from 'vue'
import { useRoute } from 'vue-router'
import { getCurrentUser } from '../utils/storage'
import TopNav from './TopNav.vue'

const TOP_NAV_HEIGHT = 100

const route = useRoute()
const isPlain = computed(() => Boolean(route.meta?.plainLayout))

const headerOffset = ref(TOP_NAV_HEIGHT)
const rightSidebarWidth = ref(0)

const setRightSidebarWidth = (width) => {
  const numericWidth = Number(width)
  rightSidebarWidth.value = Number.isFinite(numericWidth) ? Math.max(0, numericWidth) : 0
}

provide('layoutHeaderOffset', headerOffset)
provide('layoutSetRightSidebarWidth', setRightSidebarWidth)

const keepAliveUserKey = computed(() => {
  // 绑定 route，保证登录后路由切换可触发重算
  const _path = route.path
  void _path
  const username = String(getCurrentUser()?.username || '').trim()
  return username || 'anonymous'
})

const getKeepAliveKey = (r) => `${keepAliveUserKey.value}:${r.name || r.path}`

const layoutStyle = computed(() => ({
  '--header-offset': `${isPlain.value ? 0 : headerOffset.value}px`,
  '--right-sidebar-width': `${isPlain.value ? 0 : rightSidebarWidth.value}px`,
}))
</script>

<template>
  <div class="layout" :class="{ plain: isPlain }" :style="layoutStyle">
    <TopNav v-if="!isPlain" />
    <div class="layout-container" :class="{ plain: isPlain }">
      <main class="main-content" :class="{ plain: isPlain }">
        <router-view v-slot="{ Component, route: r }">
          <keep-alive>
            <component :is="Component" v-if="r.meta?.keepAlive" :key="getKeepAliveKey(r)" />
          </keep-alive>
          <component :is="Component" v-if="!r.meta?.keepAlive" :key="r.fullPath" />
        </router-view>
      </main>
    </div>
  </div>
</template>

<style scoped>
.layout {
  background-color: #f5f7fa;
  min-height: 100vh;
}

.layout.plain {
  background-color: #ffffff;
}

.layout-container {
  display: flex;
  padding-top: var(--header-offset);
  box-sizing: border-box;
}

.layout-container.plain {
  padding-top: 0;
}

.main-content {
  flex: 1;
  margin-right: var(--right-sidebar-width);
  padding: 20px 12px;
  box-sizing: border-box;
  min-height: calc(100vh - var(--header-offset));
  transition: margin-right 0.26s ease;
}

.main-content.plain {
  margin-right: 0;
  padding: 0;
  min-height: 100vh;
}
</style>
