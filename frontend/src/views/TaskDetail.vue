<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import SpectrumDetailCard from '../components/SpectrumDetailCard.vue'
import { fetchStatas } from '../api/statas'
import { buildBatchFileCards, buildTitleMapFromRows } from '../utils/statasBatch'

const route = useRoute()
const taskId = computed(() => String(route.params.taskId || ''))

document.title = taskId.value || 'task-detail'

const loading = reactive({
  page: false,
})

const isCollapsed = ref(false)
const keyword = ref('')
const pageSize = ref(12)
const currentPage = ref(1)
const fileCards = ref([])
const collapseActiveFile = ref('')
const activeTitle = ref('')
const activeFileTitleKey = ref('')

const normalMap = ref({})
const advancedMap = ref({})

const filteredFileCards = computed(() => {
  const query = keyword.value.trim()
  if (!query) return fileCards.value
  return fileCards.value.filter((item) => (item.fileName || '').includes(query))
})

const total = computed(() => filteredFileCards.value.length)
const pagedFileCards = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredFileCards.value.slice(start, start + pageSize.value)
})

const activeFile = computed(() => {
  const key = String(collapseActiveFile.value || '').trim()
  if (!key) return null
  return fileCards.value.find((item) => String(item.fileKey || '').trim() === key) || null
})

const cardItems = computed(() => {
  if (!activeTitle.value) return []
  const key = activeFileTitleKey.value || activeTitle.value
  return [{
    title: activeTitle.value,
    normalItem: normalMap.value[key] || {},
    advancedItem: advancedMap.value[key] || {},
  }]
})

const pickDefaultTitleByFile = (file) => {
  const firstTitle = file?.spectra?.[0]?.title
  return firstTitle || ''
}

const selectFile = (fileKey) => {
  collapseActiveFile.value = fileKey || ''
  const picked = fileCards.value.find((item) => item.fileKey === collapseActiveFile.value) || null
  const title = pickDefaultTitleByFile(picked)
  activeTitle.value = title
  activeFileTitleKey.value = title && picked ? `${picked.fileKey}::${title}` : ''
}

const loadTaskData = async () => {
  loading.page = true

  const [normalResp, advancedResp] = await Promise.allSettled([
    fetchStatas({ taskId: taskId.value, resultType: 'normal' }),
    fetchStatas({ taskId: taskId.value, resultType: 'advanced' }),
  ])

  const normalTaskStatas = normalResp.status === 'fulfilled' ? normalResp.value?.data?.data || {} : {}
  const advancedTaskStatas = advancedResp.status === 'fulfilled' ? advancedResp.value?.data?.data || {} : {}

  const normalList = normalTaskStatas?.['碎裂树文件统计']?.['有效碎裂树根节点信息'] || []
  const advancedList = advancedTaskStatas?.['碎裂树文件统计']?.['有效碎裂树根节点信息'] || []

  if (normalResp.status === 'rejected') {
    const msg = normalResp.reason?.response?.data?.detail || normalResp.reason?.message || 'Failed to load normal results'
    ElMessage.error(msg)
  }
  if (advancedResp.status === 'rejected') {
    const msg = advancedResp.reason?.response?.data?.detail || advancedResp.reason?.message || 'Failed to load advanced results'
    ElMessage.warning(msg)
  }

  const normalTitleMap = buildTitleMapFromRows(normalList)
  const advancedTitleMap = buildTitleMapFromRows(advancedList)

  fileCards.value = await buildBatchFileCards({
    taskStatas: normalTaskStatas,
    resultType: 'normal',
    fetchStatasByPath: (path) => fetchStatas({ path }),
  })

  const fileNameByTitle = {}
  fileCards.value.forEach((file) => {
    ;(file.spectra || []).forEach((row) => {
      if (row?.title) {
        fileNameByTitle[row.title] = file.fileKey
      }
    })
  })

  const normalCompositeMap = {}
  const advancedCompositeMap = {}
  Object.keys(normalTitleMap).forEach((title) => {
    const fileKey = fileNameByTitle[title]
    if (fileKey) normalCompositeMap[`${fileKey}::${title}`] = normalTitleMap[title]
  })
  Object.keys(advancedTitleMap).forEach((title) => {
    const fileKey = fileNameByTitle[title]
    if (fileKey) advancedCompositeMap[`${fileKey}::${title}`] = advancedTitleMap[title]
  })

  normalMap.value = { ...normalTitleMap, ...normalCompositeMap }
  advancedMap.value = { ...advancedTitleMap, ...advancedCompositeMap }

  selectFile(fileCards.value[0]?.fileKey || '')
  loading.page = false
}

const handleCollapseChange = (fileKey) => {
  selectFile(fileKey)
  nextTick(() => {
    const root = titleListRef.value?.$el || titleListRef.value
    if (!root) return
    const activeItem = root.querySelector('.el-collapse-item.is-active')
    if (activeItem && typeof activeItem.scrollIntoView === 'function') {
      activeItem.scrollIntoView({ block: 'start', behavior: 'smooth' })
    }
  })
}

const handlePickSpectrum = (title) => {
  activeTitle.value = title || ''
  activeFileTitleKey.value = activeTitle.value && collapseActiveFile.value ? `${collapseActiveFile.value}::${activeTitle.value}` : ''
}

const removeCard = (title) => {
  if (!title) return
  if (activeTitle.value === title) {
    activeTitle.value = ''
    activeFileTitleKey.value = ''
  }
}

const resetPage = () => {
  currentPage.value = 1
}

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

onMounted(() => {
  loadTaskData()
})
</script>

<template>
  <div class="task-detail-page" v-loading="loading.page">
    <aside class="sidebar" :class="{ collapsed: isCollapsed }" :style="{ width: isCollapsed ? '48px' : '360px' }">
      <div class="sidebar-content" v-if="!isCollapsed">
        <div class="panel-header">
          <strong>File list</strong>
        </div>

        <el-input v-model="keyword" placeholder="Search file name" clearable @input="resetPage" />

        <el-collapse ref="titleListRef" v-model="collapseActiveFile" accordion class="title-list" @change="handleCollapseChange">
          <el-collapse-item
            v-for="(file, idx) in pagedFileCards"
            :key="file.fileKey"
            :name="file.fileKey"
            class="title-item"
          >
            <template #title>
              <div class="title-row">
                <span class="title-index">#{{ (currentPage - 1) * pageSize + idx + 1 }}</span>
                <span class="title-text">{{ file.fileName }}</span>
                <span class="checkmark" v-if="collapseActiveFile === file.fileKey">✓</span>
              </div>
            </template>
            <div class="file-spectra-list">
              <button
                v-for="row in file.spectra"
                :key="`${file.fileKey}-${row.title}`"
                type="button"
                class="spectrum-btn"
                :class="{ active: activeTitle === row.title }"
                @click="handlePickSpectrum(row.title)"
              >
                {{ row.title }}
              </button>
            </div>
          </el-collapse-item>
        </el-collapse>

        <div class="similarity-panel">
          <div class="sim-header">
            <strong>File selection</strong>
            <span v-if="activeFile" class="sim-key">{{ activeFile.fileName }}</span>
          </div>
          <div class="sim-body" v-if="activeTitle">
            <div class="sim-hint">当前谱图：{{ activeTitle }}</div>
          </div>
          <div class="sim-body" v-else>
            <div class="sim-hint">Select one spectrum from current file card.</div>
          </div>
        </div>

        <el-pagination
          small
          layout="prev, pager, next"
          :total="total"
          :page-size="pageSize"
          v-model:current-page="currentPage"
          class="pager"
        />
      </div>

      <button class="toggle-rail" type="button" @click="toggleSidebar">{{ isCollapsed ? '>>' : '<<' }}</button>
    </aside>

    <main class="content-area">
      <el-empty v-if="!cardItems.length" description="Select one file card and spectrum from the sidebar accordion." />

      <div v-else class="cards">
        <div v-for="item in cardItems" :key="item.title" class="card-slot">
          <SpectrumDetailCard
            :task-id="taskId"
            :title="item.title"
            :normal-item="item.normalItem"
            :advanced-item="item.advancedItem"
            @close="removeCard"
          />
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
.task-detail-page {
  display: flex;
  min-height: 100vh;
  box-sizing: border-box;
  background: #f5f7fb;
}

.sidebar {
  position: relative;
  transition: width 0.2s ease;
  background: #ffffff;
  border-right: 1px solid #e5e7eb;
  box-shadow: 0 8px 20px rgba(17, 24, 39, 0.06);
  min-width: 48px;
  display: flex;
}

.sidebar-content {
  flex: 1;
  padding: 12px 60px 12px 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  box-sizing: border-box;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.title-list {
  overflow: auto;
  flex: 1;
  min-height: 0;
  max-height: calc(100vh - 260px);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.file-spectra-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 220px;
  overflow: auto;
  padding-right: 2px;
}

.similarity-panel {
  margin-top: 8px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 10px;
  background: #f9fbff;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}

.sim-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  font-size: 13px;
  color: #1f2937;
}

.sim-key {
  font-family: 'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace;
  font-size: 12px;
  color: #4b5563;
}

.sim-body {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.sim-hint {
  font-size: 12px;
  color: #6b7280;
}

.spectrum-btn {
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  background: #fff;
  color: #303133;
  font-size: 12px;
  text-align: left;
  padding: 6px 8px;
  cursor: pointer;
}

.spectrum-btn:hover {
  border-color: #409eff;
  background: #f5f9ff;
}

.spectrum-btn.active {
  border-color: #409eff;
  background: #ecf5ff;
  color: #1f2937;
}

.title-item {
  border: 1px solid #ebeef5;
  border-radius: 6px;
  background: #ffffff;
}

.title-row {
  display: grid;
  grid-template-columns: 70px 1fr 28px;
  align-items: center;
  gap: 8px;
  width: 100%;
  font-family: 'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace;
}

.title-index {
  font-weight: 700;
  color: #409eff;
  font-size: 13px;
}

.title-text {
  word-break: break-all;
  font-size: 13px;
  line-height: 1.5;
  color: #303133;
}

.checkmark {
  text-align: right;
  font-size: 14px;
  color: #303133;
}

.pager {
  justify-content: center;
  margin-top: 4px;
}

.toggle-rail {
  position: absolute;
  top: 0;
  right: 0;
  width: 48px;
  height: 100%;
  border: none;
  background: linear-gradient(180deg, #f2f6fc 0%, #d9e4f4 100%);
  border-left: 1px solid #e5e7eb;
  font-size: 18px;
  font-weight: 700;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.content-area {
  flex: 1;
  padding: 20px;
  box-sizing: border-box;
  min-width: 0;
}

.cards {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}

.card-slot {
  min-width: 0;
}

@media (max-width: 1200px) {
  .task-detail-page {
    flex-direction: column;
  }

  .sidebar {
    width: 100% !important;
    box-shadow: none;
  }

  .toggle-rail {
    height: 48px;
    width: 100%;
    position: relative;
  }

  .content-area {
    padding: 16px;
  }
}
</style>
