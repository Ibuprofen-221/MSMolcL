<script setup>
import { computed, inject, onActivated, onBeforeUnmount, onDeactivated, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { checkHealth } from '../api/health'
import { getRetrieveAdvancedStatus, runRetrieveAdvanced } from '../api/retrieve'
import { downloadTaskFile } from '../api/download'
import { fetchStatas } from '../api/statas'
import { toImageUrl, visualizeSmiles } from '../api/smiles'
import { updateHistoryRecord } from '../api/history'
import { getCurrentUser, getSessionByUser, setSessionByUser } from '../utils/storage'

const retrieveForm = reactive({
  taskId: '',
  ionMode: 'pos',
})

const retrieveSummary = ref(null)
const retrieveJob = ref(null)
const statasData = ref(null)
const spectraList = ref([])

const titleQuery = ref('')
const selectedSpectrumTitle = ref('')
const smilesSidebarOpen = ref(false)
const smilesCards = ref([])

const SMILES_RAIL_WIDTH = 40
const SMILES_PANEL_WIDTH = 380
const layoutHeaderOffset = inject('layoutHeaderOffset', ref(200))
const setRightSidebarWidth = inject('layoutSetRightSidebarWidth', () => {})

const smilesSidebarWidth = computed(() => (smilesSidebarOpen.value ? SMILES_RAIL_WIDTH + SMILES_PANEL_WIDTH : SMILES_RAIL_WIDTH))
const smilesSidebarStyle = computed(() => ({
  top: `${layoutHeaderOffset.value}px`,
  height: `calc(100vh - ${layoutHeaderOffset.value}px)`,
}))

watch(smilesSidebarWidth, (width) => {
  setRightSidebarWidth(width)
}, { immediate: true })

const collapseActive = ref([])
const pageSize = 10
const currentPage = ref(1)

const loading = reactive({
  health: false,
  retrieve: false,
  statas: false,
  smiles: false,
})

const sessionKeys = {
  task: 'advanced.taskInfo',
  config: 'advanced.retrieveConfig',
  retrieve: 'advanced.retrieveSummary',
}

const getSessionUsername = () => String(getCurrentUser()?.username || '').trim()

const RETRIEVE_POLL_INTERVAL_MS = 2000
const RETRIEVE_POLL_TIMEOUT_MS = 10 * 60 * 1000
let stopRetrievePolling = false

const effectiveCount = computed(() => spectraList.value.length)
const canDownloadStatas = computed(() => Boolean(retrieveForm.taskId && retrieveJob.value?.status === 'success'))
const pagedSpectra = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return spectraList.value.slice(start, start + pageSize)
})

const smilesSlots = computed(() => {
  const cards = smilesCards.value.slice(0, 10)
  const slots = [...cards]
  while (slots.length < 10) {
    slots.push({ index: slots.length + 1, smiles: '', image_url: '', image_src: '', status: 'empty' })
  }
  return slots
})

function mapStatasToList(statas) {
  const list = statas?.['碎裂树文件统计']?.['有效碎裂树根节点信息'] || []
  spectraList.value = list.map((item) => ({
    title: item.title,
    mz: item.mz,
    adduct: item.adduct,
    result: item['检索结果'] || null,
  }))
  currentPage.value = 1
  collapseActive.value = []
  statasData.value = statas || null
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const saveHistory = async (payload) => {
  if (!retrieveForm.taskId) return
  try {
    await updateHistoryRecord({ task_id: retrieveForm.taskId, ...payload })
  } catch (error) {
    console.error('history update failed', error)
  }
}

const copyTaskId = async () => {
  if (!retrieveForm.taskId) return
  try {
    await navigator.clipboard.writeText(retrieveForm.taskId)
    ElMessage.success('Task ID copied')
  } catch (error) {
    ElMessage.error('Copy failed')
  }
}

onMounted(() => {
  const savedTask = getSessionByUser(sessionKeys.task, getSessionUsername())
  if (savedTask?.task_id) retrieveForm.taskId = savedTask.task_id

  const savedConfig = getSessionByUser(sessionKeys.config, getSessionUsername())
  if (savedConfig?.ion_mode && ['pos', 'neg'].includes(savedConfig.ion_mode)) {
    retrieveForm.ionMode = savedConfig.ion_mode
  }

  const savedRetrieve = getSessionByUser(sessionKeys.retrieve, getSessionUsername())
  if (savedRetrieve) {
    retrieveSummary.value = savedRetrieve?.result || null
    retrieveJob.value = {
      job_id: savedRetrieve?.job_id || '',
      status: savedRetrieve?.status || '',
      error: savedRetrieve?.error || null,
    }
  }
})

onActivated(() => {
  setRightSidebarWidth(smilesSidebarWidth.value)
})

onDeactivated(() => {
  setRightSidebarWidth(0)
})

onBeforeUnmount(() => {
  stopRetrievePolling = true
  setRightSidebarWidth(0)
})

const handleHealthCheck = async () => {
  loading.health = true
  try {
    await checkHealth()
    ElMessage.success('Backend service is healthy')
  } catch (error) {
    ElMessage.error(error?.response?.data?.message || 'Backend service unavailable')
  } finally {
    loading.health = false
  }
}

const handleLoadTask = async () => {
  if (!retrieveForm.taskId?.trim()) {
    ElMessage.error('Please input task ID')
    return
  }
  loading.statas = true
  try {
    setSessionByUser(sessionKeys.task, { task_id: retrieveForm.taskId.trim() }, getSessionUsername())
    const resp = await fetchStatas({ taskId: retrieveForm.taskId.trim(), resultType: 'normal' })
    const data = resp?.data?.data
    if (data) mapStatasToList(data)
    ElMessage.success('Task loaded')
  } catch (error) {
    const msg = error?.response?.data?.message || error?.message || 'Failed to load task'
    ElMessage.error(msg)
  } finally {
    loading.statas = false
  }
}

const refreshAdvancedStatas = async () => {
  if (!retrieveForm.taskId) return
  loading.statas = true
  try {
    const resp = await fetchStatas({ taskId: retrieveForm.taskId, resultType: 'advanced' })
    const data = resp?.data?.data
    if (data) mapStatasToList(data)
  } catch (error) {
    const msg = error?.response?.data?.message || error?.message || 'Failed to read advanced statas'
    ElMessage.error(msg)
  } finally {
    loading.statas = false
  }
}

const pollRetrieveResult = async (jobId) => {
  const startedAt = Date.now()
  while (!stopRetrievePolling) {
    if (Date.now() - startedAt > RETRIEVE_POLL_TIMEOUT_MS) {
      throw new Error('Search task timed out, please retry later')
    }

    const statusResp = await getRetrieveAdvancedStatus(jobId)
    const statusData = statusResp?.data?.data
    if (!statusData) throw new Error('Failed to read search status')

    retrieveJob.value = {
      job_id: statusData.job_id,
      status: statusData.status,
      error: statusData.error || null,
    }

    if (statusData.status === 'success') {
      retrieveSummary.value = statusData.result || null
      setSessionByUser(
        sessionKeys.retrieve,
        {
          job_id: statusData.job_id,
          status: statusData.status,
          result: retrieveSummary.value,
          error: null,
        },
        getSessionUsername(),
      )
      await saveHistory({ advanced_status: 'success' })
      return
    }

    if (statusData.status === 'failed') {
      const failedMsg = statusData.error || 'Search task failed'
      setSessionByUser(
        sessionKeys.retrieve,
        {
          job_id: statusData.job_id,
          status: statusData.status,
          result: null,
          error: failedMsg,
        },
        getSessionUsername(),
      )
      await saveHistory({ advanced_status: 'failed' })
      throw new Error(failedMsg)
    }

    await sleep(RETRIEVE_POLL_INTERVAL_MS)
  }
}

const handleRetrieve = async () => {
  if (!retrieveForm.taskId?.trim()) {
    ElMessage.error('Please input task ID first')
    return
  }
  if (!['pos', 'neg'].includes(retrieveForm.ionMode)) {
    ElMessage.error('Ion mode is invalid')
    return
  }

  retrieveForm.taskId = retrieveForm.taskId.trim()
  setSessionByUser(sessionKeys.task, { task_id: retrieveForm.taskId }, getSessionUsername())
  setSessionByUser(sessionKeys.config, { ion_mode: retrieveForm.ionMode }, getSessionUsername())

  stopRetrievePolling = false
  loading.retrieve = true
  retrieveSummary.value = null

  try {
    const resp = await runRetrieveAdvanced({ task_id: retrieveForm.taskId, ion_mode: retrieveForm.ionMode })
    const submitData = resp?.data?.data
    const jobId = submitData?.job_id
    if (!jobId) throw new Error('Search submission failed: missing job_id')

    retrieveJob.value = {
      job_id: jobId,
      status: submitData?.status || 'running',
      error: null,
    }
    setSessionByUser(
      sessionKeys.retrieve,
      {
        job_id: jobId,
        status: retrieveJob.value.status,
        result: null,
        error: null,
      },
      getSessionUsername(),
    )

    await saveHistory({ advanced_status: 'pending' })
    ElMessage.success('Advanced search submitted and running')
    await pollRetrieveResult(jobId)
    ElMessage.success('Advanced search completed')
    await refreshAdvancedStatas()
  } catch (error) {
    const msg = error?.response?.data?.message || error?.message || 'Search failed'
    ElMessage.error(msg)
  } finally {
    loading.retrieve = false
  }
}

const expandAll = () => {
  if (collapseActive.value.length === spectraList.value.length) collapseActive.value = []
  else collapseActive.value = spectraList.value.map((item) => item.title)
}

const getResultTopEntries = (result) => {
  const rows = Array.isArray(result?.result_top100) ? result.result_top100 : []
  if (rows.length) return rows
  const smilesList = result?.top100 || []
  const scores = result?.top100_score || []
  return smilesList.map((smiles, idx) => ({
    rank: idx + 1,
    score: scores[idx] ?? '',
    smiles,
  }))
}

const top10Rows = (result) => {
  return getResultTopEntries(result).slice(0, 10).map((row) => ({
    smiles: row.smiles || '',
    score: row.score ?? '',
  }))
}

const findSpectrumByTitle = (title) => {
  const keyword = (title || '').trim()
  if (!keyword) return null
  return spectraList.value.find((item) => (item.title || '').trim() === keyword) || null
}

const handleGenerateSmilesImages = async () => {
  if (!spectraList.value.length) {
    ElMessage.error('No spectra available to search')
    return
  }
  const keyword = (titleQuery.value || '').trim()
  if (!keyword) {
    ElMessage.error('Please enter spectrum title')
    return
  }
  const target = findSpectrumByTitle(keyword)
  if (!target) {
    ElMessage.error('Title not found, please use a valid title from results')
    return
  }

  const top10Smiles = getResultTopEntries(target.result).slice(0, 10).map((item) => item.smiles).filter(Boolean)
  if (!top10Smiles.length) {
    ElMessage.error('This spectrum has no top10 smiles results')
    return
  }

  loading.smiles = true
  try {
    const resp = await visualizeSmiles(top10Smiles)
    const results = resp?.data?.data?.results || []
    const mapped = top10Smiles.map((smiles, idx) => {
      const row = results[idx] || {}
      const status = row.status || 'failed'
      const imageUrl = row.image_url || ''
      return {
        index: idx + 1,
        smiles,
        image_url: imageUrl,
        image_src: status === 'ready' ? toImageUrl(imageUrl) : '',
        status,
      }
    })

    smilesCards.value = mapped
    selectedSpectrumTitle.value = target.title
    smilesSidebarOpen.value = true

    const failedCount = mapped.filter((item) => item.status !== 'ready').length
    if (failedCount > 0) ElMessage.warning(`Generation finished, ${failedCount} failed`)
    else ElMessage.success('SMILES images generated')
  } catch (error) {
    const msg = error?.response?.data?.message || error?.message || 'Failed to generate SMILES images'
    ElMessage.error(msg)
  } finally {
    loading.smiles = false
  }
}

const toggleSmilesSidebar = () => {
  smilesSidebarOpen.value = !smilesSidebarOpen.value
}

const handleDownloadStatas = async () => {
  if (!canDownloadStatas.value) {
    ElMessage.error('Please finish the search first')
    return
  }

  try {
    const resp = await downloadTaskFile({ taskId: retrieveForm.taskId, filename: 'statas_advanced.json' })
    const blob = new Blob([resp.data], { type: 'application/json;charset=utf-8' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `statas_advanced_${retrieveForm.taskId}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    ElMessage.success('Result download started')
  } catch (error) {
    const msg = error?.response?.data?.message || error?.message || 'Download failed'
    ElMessage.error(msg)
  }
}
</script>

<template>
  <div class="page">
    <el-card class="big-card">
      <section class="module-section">
        <h3 class="module-title">HEALTH CHECK</h3>
        <div class="module-body">
          <el-button class="btn-unified" :loading="loading.health" @click="handleHealthCheck">Check backend status</el-button>
        </div>
      </section>

      <el-divider class="module-divider" />

      <section class="module-section">
        <h3 class="module-title">ADVANCED SEARCH BY TASK ID</h3>
        <div class="module-body">
          <div class="form-row">
            <el-input v-model="retrieveForm.taskId" placeholder="Input existing task ID" style="width: 360px" clearable />
            <el-button @click="copyTaskId" :disabled="!retrieveForm.taskId">Copy</el-button>
            <el-button class="btn-unified" :loading="loading.statas" @click="handleLoadTask">Load task</el-button>
          </div>
          <div class="form-row" style="margin-top: 12px">
            <el-select v-model="retrieveForm.ionMode" placeholder="Select ion mode" style="width: 180px">
              <el-option label="Positive (pos)" value="pos" />
              <el-option label="Negative (neg)" value="neg" />
            </el-select>
            <el-button class="btn-unified" :loading="loading.retrieve" @click="handleRetrieve">Run advanced search</el-button>
          </div>
          <div v-if="retrieveJob" class="hint">
            Status: {{ retrieveJob.status }}
            <span v-if="retrieveJob.error">| Error: {{ retrieveJob.error }}</span>
          </div>
          <div class="actions download-actions">
            <el-button type="warning" :disabled="!canDownloadStatas" @click="handleDownloadStatas">Export advanced results</el-button>
          </div>
        </div>
      </section>

      <section v-if="statasData" class="module-section">
        <el-divider class="module-divider" />
        <h3 class="module-title">SPECTRA SUMMARY</h3>
        <div class="module-body">
          <el-descriptions border :column="1" class="read-info-desc">
            <el-descriptions-item label="Valid spectra count">
              <span class="metric">{{ effectiveCount }}</span>
            </el-descriptions-item>
          </el-descriptions>
          <el-table :data="pagedSpectra" stripe highlight-current-row style="width: 100%; margin-top: 12px" :empty-text="'No spectra data'">
            <el-table-column prop="title" label="Spectrum title" align="center" />
            <el-table-column prop="mz" label="Precursor m/z" align="center" />
            <el-table-column prop="adduct" label="Adduct" align="center" />
          </el-table>
          <div v-if="effectiveCount > pageSize" class="pagination">
            <el-pagination background layout="prev, pager, next" :total="effectiveCount" :page-size="pageSize" v-model:current-page="currentPage" />
          </div>
        </div>
      </section>

      <section v-if="spectraList.length" class="module-section">
        <el-divider class="module-divider" />
        <h3 class="module-title">ADVANCED SEARCH RESULTS (TOP 10)</h3>
        <div class="module-body">
          <div class="result-toolbar">
            <div class="title-search-row">
              <el-input v-model="titleQuery" placeholder="Enter spectrum title (exact)" clearable class="title-input" />
              <el-button type="primary" :loading="loading.smiles" @click="handleGenerateSmilesImages">Generate top10 SMILES images</el-button>
            </div>
            <el-button size="small" @click="expandAll">{{ collapseActive.length === spectraList.length ? 'Collapse all' : 'Expand all' }}</el-button>
          </div>
          <el-collapse v-model="collapseActive" class="collapse">
            <el-collapse-item v-for="item in spectraList" :key="item.title" :name="item.title">
              <template #title><strong>Spectrum: {{ item.title }}</strong></template>
              <div class="panel-title">{{ item.title }} - Search Results (Top 10)</div>
              <el-table
                v-if="item.result && top10Rows(item.result).length"
                :data="top10Rows(item.result)"
                size="small"
                style="width: 100%"
                :header-cell-style="{ background: '#f0f9ff', fontWeight: '600' }"
              >
                <el-table-column prop="smiles" label="SMILES" />
                <el-table-column prop="score" label="Score" align="right">
                  <template #default="scope">{{ scope.row.score ? Number(scope.row.score).toFixed(4) : '' }}</template>
                </el-table-column>
              </el-table>
              <div v-else class="no-data">No matched results</div>
            </el-collapse-item>
          </el-collapse>
        </div>
      </section>
    </el-card>

    <aside class="smiles-sidebar" :class="{ open: smilesSidebarOpen }" :style="smilesSidebarStyle">
      <button class="smiles-rail" type="button" :aria-label="smilesSidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'" @click="toggleSmilesSidebar">
        {{ smilesSidebarOpen ? '>>' : '<<' }}
      </button>
      <div class="smiles-panel">
        <div class="smiles-sidebar-header">
          <div class="smiles-sidebar-title">Molecule images</div>
          <div class="smiles-sidebar-subtitle">{{ selectedSpectrumTitle ? `title: ${selectedSpectrumTitle}` : 'No title selected' }}</div>
        </div>
        <div class="smiles-list">
          <div v-for="slot in smilesSlots" :key="`smiles-slot-${slot.index}`" class="smiles-item">
            <div class="smiles-item-title">#{{ slot.index }} {{ slot.smiles || '(empty)' }}</div>
            <div v-if="slot.status === 'ready' && slot.image_src" class="smiles-image-wrap">
              <img :src="slot.image_src" :alt="slot.smiles" class="smiles-image" />
            </div>
            <div v-else-if="slot.status === 'failed'" class="smiles-failed">×</div>
            <div v-else class="smiles-empty">-</div>
          </div>
        </div>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.page {
  width: 98%;
  margin: 0 auto;
}

.big-card {
  border-radius: 4px;
}

.module-section {
  width: 100%;
}

.module-divider {
  margin: 16px 0;
}

.module-title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: #000000;
  letter-spacing: 0.2px;
}

.module-body {
  margin-top: 10px;
  font-size: 12px;
  font-weight: 400;
}

.actions {
  margin-top: 12px;
  display: flex;
  gap: 12px;
  align-items: center;
}

.download-actions {
  justify-content: flex-end;
}

.hint {
  margin-top: 12px;
  color: #606266;
  line-height: 1.5;
  font-size: 12px;
}

.form-row {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.metric {
  font-size: 18px;
  color: #409eff;
  font-weight: 700;
}

.pagination {
  margin-top: 12px;
  display: flex;
  justify-content: center;
}

.collapse {
  margin-top: 12px;
}

.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 8px;
}

.no-data {
  color: #909399;
  padding: 8px 0;
  font-size: 12px;
}

.result-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.title-search-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.title-input {
  width: 360px;
}

.btn-unified {
  border-radius: 0 !important;
  border: none !important;
  color: #000000 !important;
  background: #3e7ab6 !important;
}

.btn-unified:hover,
.btn-unified:focus,
.btn-unified:active {
  border: none !important;
  color: #000000 !important;
  background: #3e7ab6 !important;
}

.smiles-sidebar {
  position: fixed;
  right: 0;
  width: 40px;
  background: #ffffff;
  border-left: 1px solid #e6e6e6;
  box-shadow: -4px 0 14px rgba(0, 0, 0, 0.08);
  transition: width 0.25s ease, top 0.28s ease, height 0.28s ease;
  z-index: 1000;
  display: flex;
  overflow: hidden;
}

.smiles-sidebar.open {
  width: 420px;
}

.smiles-rail {
  width: 40px;
  height: 100%;
  border: none;
  border-right: 1px solid #ebeef5;
  background: linear-gradient(180deg, #f2f6fc 0%, #d9e4f4 100%);
  color: #000;
  font-size: 22px;
  font-weight: 700;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.smiles-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  background: #ffffff;
}

.smiles-sidebar:not(.open) .smiles-panel {
  opacity: 0;
  transform: translateX(12px);
  pointer-events: none;
}

.smiles-sidebar-header {
  padding: 12px;
  border-bottom: 1px solid #ebeef5;
}

.smiles-sidebar-title {
  font-weight: 700;
  color: #303133;
}

.smiles-sidebar-subtitle {
  margin-top: 4px;
  color: #606266;
  font-size: 12px;
  word-break: break-all;
}

.smiles-list {
  padding: 0 12px;
  overflow-y: auto;
  height: 100%;
}

.smiles-item {
  padding: 10px 0;
  border-bottom: 1px solid #ebeef5;
}

.smiles-item-title {
  color: #303133;
  font-size: 12px;
  line-height: 1.4;
  margin-bottom: 8px;
  word-break: break-all;
}

.smiles-image-wrap {
  width: 100%;
  background: #f8fafc;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  overflow: hidden;
}

.smiles-image {
  width: 100%;
  height: 160px;
  object-fit: contain;
  display: block;
}

.smiles-failed,
.smiles-empty {
  height: 160px;
  border: 1px dashed #dcdfe6;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #909399;
  background: #fafafa;
}

.smiles-failed {
  color: #f56c6c;
  font-size: 42px;
  font-weight: 700;
}
</style>
