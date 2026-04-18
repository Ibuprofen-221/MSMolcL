<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import SpectrumDetailCard from '../components/SpectrumDetailCard.vue'
import { fetchStatas } from '../api/statas'

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
const allTitles = ref([])
const activeTitles = ref([])

const normalMap = ref({})
const advancedMap = ref({})
const pairSimilarityMap = ref({})

const filteredTitles = computed(() => {
  const query = keyword.value.trim()
  if (!query) return allTitles.value
  return allTitles.value.filter((item) => item.includes(query))
})

const total = computed(() => filteredTitles.value.length)
const pagedTitles = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredTitles.value.slice(start, start + pageSize.value)
})

const selectedPairKey = computed(() => {
  if (activeTitles.value.length !== 2) return ''
  const sortedTitles = [...activeTitles.value].sort()
  return `${sortedTitles[0]}-${sortedTitles[1]}`
})

const selectedSimilarity = computed(() => {
  const key = selectedPairKey.value
  if (!key) return null
  const value = pairSimilarityMap.value[key]
  return typeof value === 'number' ? value : null
})

const formatSimilarity = (value) => {
  if (value === null || value === undefined) return 'No similarity data'
  const num = Number(value)
  if (Number.isNaN(num)) return 'No similarity data'
  return num.toFixed(4)
}

const cardItems = computed(() =>
  activeTitles.value.map((title) => ({
    title,
    normalItem: normalMap.value[title] || {},
    advancedItem: advancedMap.value[title] || {},
  }))
)

const mergeTitleList = (normalList, advancedList) => {
  const ordered = []
  const set = new Set()
  normalList.forEach((item) => {
    if (!set.has(item.title)) {
      ordered.push(item.title)
      set.add(item.title)
    }
  })
  advancedList.forEach((item) => {
    if (!set.has(item.title)) {
      ordered.push(item.title)
      set.add(item.title)
    }
  })
  return ordered
}

const toTitleMap = (list) => {
  const map = {}
  list.forEach((item) => {
    if (item?.title) {
      map[item.title] = item
    }
  })
  return map
}

const loadTaskData = async () => {
  loading.page = true

  const [normalResp, advancedResp] = await Promise.allSettled([
    fetchStatas({ taskId: taskId.value, resultType: 'normal' }),
    fetchStatas({ taskId: taskId.value, resultType: 'advanced' }),
  ])

  const normalList =
    normalResp.status === 'fulfilled'
      ? normalResp.value?.data?.data?.['碎裂树文件统计']?.['有效碎裂树根节点信息'] || []
      : []

  const advancedList =
    advancedResp.status === 'fulfilled'
      ? advancedResp.value?.data?.data?.['碎裂树文件统计']?.['有效碎裂树根节点信息'] || []
      : []

  const pairSimMap =
    normalResp.status === 'fulfilled'
      ? normalResp.value?.data?.data?.['Spectrum Similarity'] || normalResp.value?.data?.data?.['谱图相似度'] || {}
      : {}

  if (normalResp.status === 'rejected') {
    const msg = normalResp.reason?.response?.data?.detail || normalResp.reason?.message || 'Failed to load normal results'
    ElMessage.error(msg)
  }
  if (advancedResp.status === 'rejected') {
    const msg = advancedResp.reason?.response?.data?.detail || advancedResp.reason?.message || 'Failed to load advanced results'
    ElMessage.warning(msg)
  }

  normalMap.value = toTitleMap(normalList)
  advancedMap.value = toTitleMap(advancedList)
  pairSimilarityMap.value = pairSimMap
  allTitles.value = mergeTitleList(normalList, advancedList)

  loading.page = false
}

const addCard = (title) => {
  if (!title) return
  if (activeTitles.value.includes(title)) return
  if (activeTitles.value.length >= 2) {
    activeTitles.value.shift()
  }
  activeTitles.value.push(title)
}

const removeCard = (title) => {
  activeTitles.value = activeTitles.value.filter((item) => item !== title)
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
          <strong>Spectrum list</strong>
        </div>

        <el-input v-model="keyword" placeholder="Search spectrum title" clearable @input="resetPage" />

        <div class="title-list">
          <div
            v-for="(title, idx) in pagedTitles"
            :key="title"
            class="title-item"
            @click="addCard(title)"
          >
            <div class="title-row">
              <span class="title-index">#{{ (currentPage - 1) * pageSize + idx + 1 }}</span>
              <span class="title-text">{{ title }}</span>
              <span class="checkmark" v-if="activeTitles.includes(title)">✓</span>
            </div>
          </div>
        </div>

        <div class="similarity-panel">
          <div class="sim-header">
            <strong>Cosine similarity</strong>
            <span v-if="selectedPairKey" class="sim-key">{{ selectedPairKey }}</span>
          </div>
          <div class="sim-body" v-if="activeTitles.length === 2">
            <div class="sim-value">{{ formatSimilarity(selectedSimilarity) }}</div>
            <div class="sim-hint">Based on normal-search vectors</div>
          </div>
          <div class="sim-body" v-else>
            <div class="sim-hint">Select two spectra to view similarity</div>
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
      <el-empty v-if="!cardItems.length" description="Select spectra from the sidebar. Up to 2 cards." />

      <div v-else class="cards" :class="{ dual: cardItems.length === 2 }">
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
  display: flex;
  flex-direction: column;
  gap: 10px;
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

.sim-value {
  font-size: 20px;
  font-weight: 700;
  color: #409eff;
  line-height: 1.1;
}

.sim-hint {
  font-size: 12px;
  color: #6b7280;
}

.title-item {
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 8px 10px;
  background: #ffffff;
  cursor: pointer;
  transition: box-shadow 0.15s ease, transform 0.12s ease;
}

.title-item:hover {
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
  transform: translateY(-1px);
}

.title-row {
  display: grid;
  grid-template-columns: 70px 1fr 28px;
  align-items: center;
  gap: 8px;
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

.cards.dual {
  grid-template-columns: repeat(2, minmax(0, 1fr));
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

  .cards.dual {
    grid-template-columns: 1fr;
  }
}
</style>
