<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import Plotly from 'plotly.js-dist-min'
import { ElMessage } from 'element-plus'
import { checkHealth } from '../api/health'
import { getRetrieveAdvancedStatus, runRetrieveAdvanced } from '../api/retrieve'
import { downloadTaskFile } from '../api/download'
import { fetchSpectrumPlot } from '../api/spectrum'
import { fetchStatas } from '../api/statas'
import { buildBatchFileCards, flattenFileCardSpectra } from '../utils/statasBatch'
import { updateHistoryRecord } from '../api/history'
import { getCurrentUser, getSessionByUser, removeSessionByUser, setSessionByUser } from '../utils/storage'

const STEP_TASK = 'task'
const STEP_FILES = 'files'
const STEP_RESULTS = 'results'

const retrieveForm = reactive({ taskId: '', ionMode: 'pos' })
const retrieveSummary = ref(null)
const retrieveJob = ref(null)
const statasData = ref(null)
const spectraList = ref([])
const fileCards = ref([])
const globalError = ref('')
const currentStep = ref(STEP_TASK)

const collapseActiveFile = ref('')
const resultActiveFile = ref('')
const resultActiveTitle = ref('')
const summaryCollapseRef = ref(null)
const resultCollapseRef = ref(null)
const resultPlotRef = ref(null)
const resultPlotLoading = ref(false)
const resultPlotError = ref('')
const resultPlotKey = ref('')
const pageSize = 10
const currentPage = ref(1)
const resultPageSize = 12
const resultCurrentPage = ref(1)

const loading = reactive({ health: false, retrieve: false, statas: false })

const sessionKeys = {
  task: 'advanced.taskInfo',
  config: 'advanced.retrieveConfig',
  retrieve: 'advanced.retrieveSummary',
}

const RETRIEVE_POLL_INTERVAL_MS = 2000
const RETRIEVE_POLL_TIMEOUT_MS = 10 * 60 * 1000
let stopRetrievePolling = false

const getSessionUsername = () => String(getCurrentUser()?.username || '').trim()
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const taskLoaded = computed(() => Boolean(retrieveForm.taskId?.trim()) && Boolean(statasData.value))
const retrieveDone = computed(() => ['success', 'partial_failed'].includes(retrieveJob.value?.status || ''))

const fileCount = computed(() => fileCards.value.length)
const pagedFileCards = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return fileCards.value.slice(start, start + pageSize)
})
const pagedResultFiles = computed(() => {
  const start = (resultCurrentPage.value - 1) * resultPageSize
  return fileCards.value.slice(start, start + resultPageSize)
})

const resultCurrentFile = computed(() => {
  const key = String(resultActiveFile.value || '').trim()
  if (!key) return null
  return fileCards.value.find((item) => String(item.fileKey || '').trim() === key) || null
})
const resultCurrentSpectra = computed(() => resultCurrentFile.value?.spectra || [])
const resultCurrentSpectrum = computed(() => {
  const key = String(resultActiveTitle.value || '').trim()
  if (!key) return null
  return resultCurrentSpectra.value.find((item) => String(item.title || '').trim() === key) || null
})
const summaryCurrentFile = computed(() => {
  const key = String(collapseActiveFile.value || '').trim()
  if (!key) return pagedFileCards.value[0] || fileCards.value[0] || null
  return fileCards.value.find((item) => String(item.fileKey || '').trim() === key) || null
})
const currentValidSpectraCount = computed(() => Number(summaryCurrentFile.value?.spectra?.length || 0))

const advancedResultEntries = computed(() => {
  const rows = Array.isArray(resultCurrentSpectrum.value?.result?.result_top100)
    ? resultCurrentSpectrum.value.result.result_top100
    : []
  return rows.map((row, idx) => ({
    rank: row?.rank ?? (idx + 1),
    score: row?.score ?? '',
    smiles: row?.smiles || '',
    formula: row?.formula || '',
    inchi_key: row?.inchi_key || '',
    generic_name: row?.generic_name || '',
    database_name: row?.database_name || '',
    database_id: row?.database_id || '',
  }))
})

const renderResultPlot = async () => {
  await nextTick()
  const title = String(resultCurrentSpectrum.value?.title || '').trim()
  if (!retrieveForm.taskId || !title) {
    resultPlotLoading.value = false
    resultPlotError.value = ''
    if (resultPlotRef.value) Plotly.purge(resultPlotRef.value)
    return
  }

  const requestKey = `${retrieveForm.taskId}::${title}`
  resultPlotKey.value = requestKey
  resultPlotLoading.value = true
  resultPlotError.value = ''

  try {
    const resp = await fetchSpectrumPlot({ taskId: retrieveForm.taskId, title })
    const payload = resp?.data?.data?.plotly_data
    if (!payload?.data || !payload?.layout) {
      throw new Error('Invalid spectrum payload')
    }
    if (resultPlotKey.value !== requestKey) return

    await nextTick()
    if (!resultPlotRef.value) return

    await Plotly.react(resultPlotRef.value, payload.data, payload.layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    })
  } catch (error) {
    if (resultPlotKey.value !== requestKey) return
    resultPlotError.value = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Failed to load spectrum plot'
    if (resultPlotRef.value) Plotly.purge(resultPlotRef.value)
  } finally {
    if (resultPlotKey.value === requestKey) resultPlotLoading.value = false
  }
}

watch(
  () => [retrieveForm.taskId, resultCurrentSpectrum.value?.title],
  () => {
    renderResultPlot()
  },
  { immediate: true, flush: 'post' }
)

const stepItems = computed(() => [
  { key: STEP_TASK, label: 'Task Load' },
  { key: STEP_FILES, label: 'Files' },
  { key: STEP_RESULTS, label: 'Results' },
])

const isStepAccessible = (step) => {
  if (step === STEP_TASK) return true
  if (step === STEP_FILES) return taskLoaded.value
  if (step === STEP_RESULTS) return retrieveDone.value
  return false
}

const isStepCompleted = (step) => {
  if (step === STEP_TASK) return taskLoaded.value
  if (step === STEP_FILES) return taskLoaded.value
  if (step === STEP_RESULTS) return retrieveDone.value
  return false
}

const canDownloadStatas = computed(() => Boolean(retrieveForm.taskId && retrieveDone.value))

const setGlobalError = (msg) => {
  globalError.value = msg || ''
  if (msg) ElMessage.error(msg)
}

const clearGlobalError = () => {
  globalError.value = ''
}

const getStepClass = (step) => {
  const isCurrent = currentStep.value === step
  const done = isStepCompleted(step)
  return { current: isCurrent, done, locked: !isStepAccessible(step) }
}

const gotoStep = (step) => {
  if (currentStep.value === step) return
  if (!isStepAccessible(step)) {
    setGlobalError('请先完成前序步骤后再进入该板块')
    return
  }
  currentStep.value = step
}

const pickResultDefaults = () => {
  resultCurrentPage.value = 1
  resultActiveFile.value = fileCards.value[0]?.fileKey || ''
  resultActiveTitle.value = ''
}

const scrollActiveCollapseToTop = (collapseRef) => {
  nextTick(() => {
    const root = collapseRef?.value?.$el || collapseRef?.value
    if (!root) return
    const activeItem = root.querySelector('.el-collapse-item.is-active')
    if (activeItem && typeof activeItem.scrollIntoView === 'function') {
      activeItem.scrollIntoView({ block: 'start', behavior: 'smooth' })
    }
  })
}

const onResultFileChange = (fileKey) => {
  resultActiveFile.value = fileKey || ''
  resultActiveTitle.value = ''
  scrollActiveCollapseToTop(resultCollapseRef)
}

const onSummaryCollapseChange = (name) => {
  collapseActiveFile.value = name || ''
  scrollActiveCollapseToTop(summaryCollapseRef)
}

async function mapStatasToList(statas, resultType = 'normal') {
  const cards = await buildBatchFileCards({
    taskStatas: statas,
    resultType,
    fetchStatasByPath: (path) => fetchStatas({ path }),
  })

  fileCards.value = cards
  spectraList.value = flattenFileCardSpectra(cards)
  currentPage.value = 1
  collapseActiveFile.value = cards[0]?.fileKey || ''
  statasData.value = statas || null
  pickResultDefaults()
}

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
    setGlobalError('Copy failed')
  }
}

onMounted(() => {
  const savedTask = getSessionByUser(sessionKeys.task, getSessionUsername())
  if (savedTask?.task_id) retrieveForm.taskId = savedTask.task_id

  const savedConfig = getSessionByUser(sessionKeys.config, getSessionUsername())
  if (savedConfig?.ion_mode && ['pos', 'neg'].includes(savedConfig.ion_mode)) retrieveForm.ionMode = savedConfig.ion_mode

  const savedRetrieve = getSessionByUser(sessionKeys.retrieve, getSessionUsername())
  if (savedRetrieve && String(savedRetrieve?.task_id || '') === String(retrieveForm.taskId || '')) {
    retrieveSummary.value = savedRetrieve?.result || null
    retrieveJob.value = {
      job_id: savedRetrieve?.job_id || '',
      status: savedRetrieve?.status || '',
      error: savedRetrieve?.error || null,
      failed_count: Number(savedRetrieve?.failed_count || 0),
      total_count: Number(savedRetrieve?.total_count || 0),
    }
  }
})

onBeforeUnmount(() => {
  stopRetrievePolling = true
  if (resultPlotRef.value) Plotly.purge(resultPlotRef.value)
})

const handleHealthCheck = async () => {
  clearGlobalError()
  loading.health = true
  try {
    await checkHealth()
    ElMessage.success('Backend service is healthy')
  } catch (error) {
    setGlobalError(error?.response?.data?.message || 'Backend service unavailable')
  } finally {
    loading.health = false
  }
}

const handleLoadTask = async () => {
  clearGlobalError()
  if (!retrieveForm.taskId?.trim()) return setGlobalError('Please input task ID')

  loading.statas = true
  try {
    retrieveForm.taskId = retrieveForm.taskId.trim()
    setSessionByUser(sessionKeys.task, { task_id: retrieveForm.taskId }, getSessionUsername())

    const resp = await fetchStatas({ taskId: retrieveForm.taskId, resultType: 'advanced' })
    const data = resp?.data?.data
    if (data) await mapStatasToList(data, 'advanced')

    retrieveSummary.value = null
    retrieveJob.value = null
    removeSessionByUser(sessionKeys.retrieve, getSessionUsername())
    currentStep.value = STEP_FILES
    ElMessage.success('Task loaded')
  } catch (error) {
    setGlobalError(error?.response?.data?.message || error?.message || 'Failed to load task')
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
    if (data) await mapStatasToList(data, 'advanced')
  } catch (error) {
    setGlobalError(error?.response?.data?.message || error?.message || 'Failed to read advanced statas')
  } finally {
    loading.statas = false
  }
}

const pollRetrieveResult = async (jobId) => {
  const startedAt = Date.now()
  while (!stopRetrievePolling) {
    if (Date.now() - startedAt > RETRIEVE_POLL_TIMEOUT_MS) throw new Error('Search task timed out, please retry later')

    const statusResp = await getRetrieveAdvancedStatus(jobId)
    const statusData = statusResp?.data?.data
    if (!statusData) throw new Error('Failed to read search status')

    retrieveJob.value = {
      job_id: statusData.job_id,
      status: statusData.status,
      error: statusData.error || null,
      failed_count: Number(statusData.failed_count || 0),
      total_count: Number(statusData.total_count || 0),
    }

    if (['success', 'partial_failed'].includes(statusData.status)) {
      retrieveSummary.value = statusData.result || null
      setSessionByUser(sessionKeys.retrieve, {
        task_id: retrieveForm.taskId,
        job_id: statusData.job_id,
        status: statusData.status,
        result: retrieveSummary.value,
        error: null,
        failed_count: Number(statusData.failed_count || 0),
        total_count: Number(statusData.total_count || 0),
      }, getSessionUsername())
      await saveHistory({ advanced_status: statusData.status === 'partial_failed' ? 'failed' : 'success' })
      return
    }

    if (statusData.status === 'failed') {
      const failedMsg = statusData.error || 'Search task failed'
      setSessionByUser(sessionKeys.retrieve, { job_id: statusData.job_id, status: statusData.status, result: null, error: failedMsg }, getSessionUsername())
      await saveHistory({ advanced_status: 'failed' })
      throw new Error(failedMsg)
    }

    await sleep(RETRIEVE_POLL_INTERVAL_MS)
  }
}

const handleRetrieve = async () => {
  clearGlobalError()
  if (!retrieveForm.taskId?.trim()) return setGlobalError('Please input task ID first')
  if (!['pos', 'neg'].includes(retrieveForm.ionMode)) return setGlobalError('Ion mode is invalid')

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

    retrieveJob.value = { job_id: jobId, status: submitData?.status || 'running', error: null }
    setSessionByUser(sessionKeys.retrieve, { task_id: retrieveForm.taskId, job_id: jobId, status: retrieveJob.value.status, result: null, error: null, failed_count: 0, total_count: 0 }, getSessionUsername())

    await saveHistory({ advanced_status: 'pending' })
    ElMessage.success('Advanced search submitted and running')
    await pollRetrieveResult(jobId)
    await refreshAdvancedStatas()
    currentStep.value = STEP_RESULTS
    ElMessage.success('Advanced search completed')
  } catch (error) {
    setGlobalError(error?.response?.data?.message || error?.message || 'Search failed')
  } finally {
    loading.retrieve = false
  }
}

const handleDownloadStatas = async () => {
  clearGlobalError()
  if (!canDownloadStatas.value) return setGlobalError('Please finish the search first')

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
    setGlobalError(error?.response?.data?.message || error?.message || 'Download failed')
  }
}
</script>

<template>
  <div class="page">
    <el-alert
      v-if="globalError"
      class="fixed-error"
      type="error"
      :closable="true"
      :title="globalError"
      @close="clearGlobalError"
      show-icon
    />

    <div class="flow-nav-wrap advanced">
      <div class="flow-line"></div>
      <button
        v-for="step in stepItems"
        :key="step.key"
        type="button"
        class="flow-step"
        :class="getStepClass(step.key)"
        :disabled="!isStepAccessible(step.key) && currentStep !== step.key"
        @click="gotoStep(step.key)"
      >
        <span class="dot"><span v-if="currentStep === step.key" class="dot-inner"></span></span>
        <span class="label">{{ step.label }}</span>
      </button>
    </div>

    <el-card class="big-card">
      <section v-show="currentStep === STEP_TASK" class="module-section">
        <h3 class="module-title">HEALTH CHECK</h3>
        <div class="module-body">
          <el-button class="btn-unified" :loading="loading.health" @click="handleHealthCheck">Check backend status</el-button>
        </div>

        <el-divider class="module-divider" />

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
            <el-button type="warning" :disabled="!canDownloadStatas" @click="handleDownloadStatas">Export advanced results</el-button>
          </div>
          <div v-if="retrieveJob" class="hint">
            Status: {{ retrieveJob.status }}
            <span v-if="retrieveJob.status === 'partial_failed'">| Failed/Total: {{ retrieveJob.failed_count || 0 }}/{{ retrieveJob.total_count || 0 }}</span>
            <span v-if="retrieveJob.error">| Error: {{ retrieveJob.error }}</span>
          </div>
        </div>
      </section>

      <section v-show="currentStep === STEP_FILES" class="module-section">
        <h3 class="module-title">FILES</h3>
        <div class="module-body">
          <el-descriptions border :column="1" class="read-info-desc">
            <el-descriptions-item label="Valid spectra count">
              <span class="metric">{{ currentValidSpectraCount }}</span>
            </el-descriptions-item>
          </el-descriptions>
          <el-collapse v-model="collapseActiveFile" accordion class="collapse" @change="onSummaryCollapseChange">
            <el-collapse-item v-for="item in pagedFileCards" :key="item.fileKey" :name="item.fileKey">
              <template #title><strong>{{ item.fileName }}</strong></template>
              <el-table :data="item.spectra" size="small" style="width: 100%" :header-cell-style="{ background: '#f0f9ff', fontWeight: '600' }">
                <el-table-column prop="title" label="Title" min-width="240" show-overflow-tooltip />
                <el-table-column prop="mz" label="Pepmass" min-width="160" align="center" />
                <el-table-column prop="adduct" label="Adduct" min-width="160" align="center" />
                <el-table-column prop="peaks" label="Peaks" min-width="160" align="center" />
              </el-table>
            </el-collapse-item>
          </el-collapse>
          <div v-if="fileCount > pageSize" class="pagination">
            <el-pagination background layout="prev, pager, next" :total="fileCount" :page-size="pageSize" v-model:current-page="currentPage" />
          </div>
        </div>
      </section>

      <section v-show="currentStep === STEP_RESULTS" class="module-section">
        <h3 class="module-title">RESULTS CHECK</h3>
        <div class="result-layout">
          <aside class="result-sidebar">
            <div class="result-sidebar-title">File list</div>
            <div class="result-file-scroll">
              <el-collapse ref="resultCollapseRef" v-model="resultActiveFile" accordion class="collapse result-collapse-scroll" @change="onResultFileChange">
                <el-collapse-item v-for="file in pagedResultFiles" :key="file.fileKey" :name="file.fileKey">
                  <template #title><strong>{{ file.fileName }}</strong></template>
                  <div class="file-spectra-list">
                    <button
                      v-for="row in file.spectra"
                      :key="`${file.fileKey}-${row.title}`"
                      type="button"
                      class="spectrum-btn"
                      :class="{ active: resultActiveTitle === row.title }"
                      @click="resultActiveTitle = row.title"
                    >
                      {{ row.title }}
                    </button>
                  </div>
                </el-collapse-item>
              </el-collapse>
            </div>
            <el-pagination
              v-if="fileCount > resultPageSize"
              small
              layout="prev, pager, next"
              :total="fileCount"
              :page-size="resultPageSize"
              v-model:current-page="resultCurrentPage"
              class="pager"
            />
          </aside>

          <main class="result-content">
            <el-empty v-if="!resultCurrentSpectrum" description="Select one spectrum from the left list" />
            <div v-else class="result-cards-scroll">
              <el-card class="result-panel-card" shadow="never">
                <template #header>
                  <div class="result-panel-header">Spectrum plot</div>
                </template>
                <div class="result-plot-wrapper" v-loading="resultPlotLoading">
                  <el-alert v-if="resultPlotError" :title="resultPlotError" type="error" :closable="false" show-icon />
                  <div ref="resultPlotRef" class="result-plot-canvas"></div>
                </div>
              </el-card>

              <el-card class="result-panel-card" shadow="never">
                <template #header>
                  <div class="result-panel-header">Spectrum information</div>
                </template>

                <el-descriptions border :column="2" class="result-info-desc">
                  <el-descriptions-item label="Title">{{ resultCurrentSpectrum?.title || '-' }}</el-descriptions-item>
                  <el-descriptions-item label="Adduct">{{ resultCurrentSpectrum?.adduct || '-' }}</el-descriptions-item>
                  <el-descriptions-item label="Precursor m/z">{{ resultCurrentSpectrum?.mz ?? '-' }}</el-descriptions-item>
                  <el-descriptions-item label="Peaks">{{ resultCurrentSpectrum?.peaks ?? 0 }}</el-descriptions-item>
                </el-descriptions>

                <div class="normal-results-block">
                  <div class="normal-results-title">Advanced search Top100 details</div>
                  <el-empty v-if="!advancedResultEntries.length" description="No advanced search results" />
                  <el-table v-else :data="advancedResultEntries" size="small" class="result-table" :header-cell-style="{ background: '#f8fafc', fontWeight: '600' }">
                    <el-table-column prop="rank" label="Rank" width="80" />
                    <el-table-column prop="score" label="Score" width="120" />
                    <el-table-column prop="smiles" label="SMILES" min-width="220" show-overflow-tooltip />
                    <el-table-column prop="formula" label="FORMULA" min-width="140" />
                    <el-table-column prop="generic_name" label="GENERIC_NAME" min-width="180" show-overflow-tooltip />
                    <el-table-column prop="database_name" label="DATABASE_NAME" min-width="160" show-overflow-tooltip />
                    <el-table-column prop="database_id" label="DATABASE_ID" min-width="160" show-overflow-tooltip />
                    <el-table-column prop="inchi_key" label="INCHI_KEY" min-width="200" show-overflow-tooltip />
                  </el-table>
                </div>
              </el-card>
            </div>
          </main>
        </div>
      </section>
    </el-card>
  </div>
</template>

<style scoped>
.page { width: 98%; margin: 0 auto; }
.big-card { border-radius: 4px; }
.module-section { width: 100%; }
.module-divider { margin: 16px 0; }
.module-title { margin: 0; font-size: 18px; font-weight: 700; color: #000; }
.module-body { margin-top: 10px; font-size: 12px; }
.fixed-error {
  position: fixed;
  top: 118px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 2000;
  width: min(920px, calc(100vw - 40px));
}
.flow-nav-wrap {
  margin: 10px 0 14px;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 14px 20px 10px;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 4px;
  position: sticky;
  top: 100px;
  z-index: 900;
}
.flow-line {
  position: absolute;
  left: 16.67%;
  right: 16.67%;
  top: 23px;
  height: 2px;
  background: #dcdfe6;
  z-index: 0;
}
.flow-step {
  position: relative;
  z-index: 1;
  border: none;
  background: transparent;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 0;
}
.flow-step:disabled { cursor: not-allowed; opacity: 0.55; }
.dot {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 2px solid #dcdfe6;
  background: #fff;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}
.flow-step.done .dot { border-color: #67c23a; background: #67c23a; }
.flow-step.current .dot { border-color: #3e7ab6; background: #3e7ab6; }
.flow-step.current .dot-inner { width: 6px; height: 6px; background: #fff; border-radius: 50%; }
.label { font-size: 13px; color: #606266; font-weight: 500; }
.flow-step.current .label { color: #1f2937; font-weight: 600; }
.flow-step.done .label { color: #303133; }

.actions { margin-top: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.hint { margin-top: 12px; color: #606266; line-height: 1.5; font-size: 12px; }
.form-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.metric { font-size: 18px; color: #409eff; font-weight: 700; }
.pagination { margin-top: 12px; display: flex; justify-content: center; }
.collapse { margin-top: 12px; }
.btn-unified { border-radius: 0 !important; border: none !important; color: #000 !important; background: #3e7ab6 !important; }
.btn-unified:hover, .btn-unified:focus, .btn-unified:active { border: none !important; color: #000 !important; background: #3e7ab6 !important; }

.result-layout { margin-top: 10px; display: grid; grid-template-columns: 320px 1fr; gap: 12px; height: calc(100vh - 240px); min-height: 560px; }
.result-sidebar { border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; padding: 10px; display: flex; flex-direction: column; height: 100%; min-height: 0; }
.result-sidebar-title { font-weight: 700; color: #1f2937; }
.result-file-scroll { margin-top: 8px; flex: 1; min-height: 0; overflow: auto; }
.result-collapse-scroll { margin-top: 0; }
.file-spectra-list { display: flex; flex-direction: column; gap: 6px; height: calc(100vh - 430px); min-height: 180px; overflow: auto; padding-right: 2px; }
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
.spectrum-btn:hover { border-color: #409eff; background: #f5f9ff; }
.spectrum-btn.active { border-color: #409eff; background: #ecf5ff; }
.pager { margin-top: 10px; justify-content: center; }
.result-content { border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; padding: 12px; height: 100%; min-height: 0; }
.result-cards-scroll { height: 100%; min-height: 0; overflow: auto; display: flex; flex-direction: column; gap: 12px; }
.result-panel-card { flex: 0 0 auto; border-radius: 8px; }
.result-panel-header { font-weight: 700; color: #1f2937; }
.result-plot-wrapper { border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; }
.result-plot-canvas { min-height: 380px; }
.result-info-desc { margin-bottom: 12px; }
.normal-results-block { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }
.normal-results-title { font-weight: 700; margin-bottom: 8px; }
.result-table { width: 100%; }

@media (max-width: 1100px) {
  .result-layout { grid-template-columns: 1fr; height: auto; }
  .result-sidebar { height: auto; }
  .file-spectra-list { height: 220px; }
}
</style>
