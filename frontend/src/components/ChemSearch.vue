<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import Plotly from 'plotly.js-dist-min'
import { ElMessage } from 'element-plus'
import { checkHealth } from '../api/health'
import { getUploadFilesStatus, uploadFiles } from '../api/upload'
import { chooseCandidates, getCandidateDatabases } from '../api/candidate'
import { getRetrieveStatus, runRetrieve } from '../api/retrieve'
import { downloadTaskFile } from '../api/download'
import { fetchSpectrumPlot } from '../api/spectrum'
import { fetchStatas } from '../api/statas'
import { buildBatchFileCards, flattenFileCardSpectra } from '../utils/statasBatch'
import { updateHistoryRecord } from '../api/history'
import { getCurrentUser, getSessionByUser, removeSessionByUser, setSessionByUser } from '../utils/storage'

const STEP_UPLOAD = 'upload'
const STEP_FILES = 'files'
const STEP_CANDIDATES = 'candidates'
const STEP_RESULTS = 'results'

const uploadForm = reactive({ mgfFiles: [], jsonFiles: [] })
const chooseForm = reactive({
  searchType: 'pubchem',
  ionMode: 'pos',
  ppmRange: 10,
  databases: ['pubchem'],
  customFile: null,
})
const paths = reactive({ taskId: '', statasPath: '', fragtreesPath: '', spectraPath: '' })

const chooseData = ref(null)
const availableDatabases = ref([])
const defaultDatabases = ref(['pubchem'])
const retrieveSummary = ref(null)
const retrieveJob = ref(null)
const statasData = ref(null)
const spectraList = ref([])
const fileCards = ref([])
const batchSummary = ref(null)
const uploadProgress = ref(0)
const uploadState = ref('idle')
const currentStep = ref(STEP_UPLOAD)
const globalError = ref('')

const mgfUploadRef = ref(null)
const jsonUploadRef = ref(null)
const customUploadRef = ref(null)
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

const loading = reactive({ health: false, upload: false, choose: false, retrieve: false, statas: false })

const sessionKeys = {
  upload: 'uploadFilesInfo',
  choose: 'retrieveConfig',
  retrieve: 'retrieveSummary',
}

const RETRIEVE_POLL_INTERVAL_MS = 2000
const RETRIEVE_POLL_TIMEOUT_MS = 60 * 60 * 1000
const UPLOAD_POLL_INTERVAL_MS = 2000
const UPLOAD_POLL_TIMEOUT_MS = 60 * 60 * 1000
let stopRetrievePolling = false
let stopUploadPolling = false

const getSessionUsername = () => String(getCurrentUser()?.username || '').trim()
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const uploadDone = computed(() => Boolean(paths.taskId) && uploadState.value === 'finished')
const chooseDone = computed(() => Boolean(chooseData.value))
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

const normalResultEntries = computed(() => {
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
  if (!paths.taskId || !title) {
    resultPlotLoading.value = false
    resultPlotError.value = ''
    if (resultPlotRef.value) Plotly.purge(resultPlotRef.value)
    return
  }

  const requestKey = `${paths.taskId}::${title}`
  resultPlotKey.value = requestKey
  resultPlotLoading.value = true
  resultPlotError.value = ''

  try {
    const resp = await fetchSpectrumPlot({ taskId: paths.taskId, title })
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
  () => [paths.taskId, resultCurrentSpectrum.value?.title],
  () => {
    renderResultPlot()
  },
  { immediate: true, flush: 'post' }
)

const stepItems = computed(() => [
  { key: STEP_UPLOAD, label: 'Upload' },
  { key: STEP_FILES, label: 'Files' },
  { key: STEP_CANDIDATES, label: 'Candidates' },
  { key: STEP_RESULTS, label: 'Results' },
])

const isStepAccessible = (step) => {
  if (step === STEP_UPLOAD) return true
  if (step === STEP_FILES) return uploadDone.value
  if (step === STEP_CANDIDATES) return uploadDone.value
  if (step === STEP_RESULTS) return retrieveDone.value
  return false
}

const isStepCompleted = (step) => {
  if (step === STEP_UPLOAD) return uploadDone.value
  if (step === STEP_FILES) return uploadDone.value
  if (step === STEP_CANDIDATES) return chooseDone.value
  if (step === STEP_RESULTS) return retrieveDone.value
  return false
}

const canDownloadStatas = computed(() => Boolean(paths.taskId && retrieveDone.value))
const canSubmitUpload = computed(() => !['finished', 'processing'].includes(uploadState.value) && !loading.upload)
const canClearUpload = computed(() => Boolean(
  uploadForm.mgfFiles.length ||
  uploadForm.jsonFiles.length ||
  chooseForm.customFile ||
  paths.taskId ||
  statasData.value ||
  batchSummary.value ||
  chooseData.value ||
  retrieveSummary.value ||
  retrieveJob.value ||
  spectraList.value.length
))
const uploadButtonText = computed(() => {
  if (uploadState.value === 'finished') return 'Finished'
  if (uploadState.value === 'processing') return 'Processing...'
  return 'Submit'
})

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
  return {
    current: isCurrent,
    done,
    locked: !isStepAccessible(step),
  }
}

const gotoStep = (step) => {
  if (currentStep.value === step) return
  if (!isStepAccessible(step)) {
    setGlobalError('请先完成前序步骤后再进入该板块')
    return
  }
  currentStep.value = step
}

const scrollActiveCollapseToTop = (collapseRef) => {
  nextTick(() => {
    const root = collapseRef?.value?.$el || collapseRef?.value
    const activeItem = root?.querySelector('.el-collapse-item.is-active')
    activeItem?.scrollIntoView?.({ block: 'start', behavior: 'smooth' })
  })
}

const pickResultDefaults = () => {
  resultCurrentPage.value = 1
  resultActiveFile.value = fileCards.value[0]?.fileKey || ''
  resultActiveTitle.value = ''
}

const handleSpectrumSelect = (title) => {
  resultActiveTitle.value = title || ''
}

const onResultFileChange = (fileKey) => {
  resultActiveFile.value = fileKey || ''
  resultActiveTitle.value = ''
  scrollActiveCollapseToTop(resultCollapseRef)
}

const onSummaryCollapseChange = (name) => {
  collapseActiveFile.value = name || ''
}

async function mapStatasToList(statas) {
  const cards = await buildBatchFileCards({
    taskStatas: statas,
    resultType: 'normal',
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
  if (!paths.taskId) return
  try {
    await updateHistoryRecord({ task_id: paths.taskId, ...payload })
  } catch (error) {
    console.error('history update failed', error)
  }
}

const copyTaskId = async () => {
  if (!paths.taskId) return
  try {
    await navigator.clipboard.writeText(paths.taskId)
    ElMessage.success('Task ID copied')
  } catch (error) {
    setGlobalError('Copy failed')
  }
}

const loadCandidateDatabases = async () => {
  try {
    const resp = await getCandidateDatabases()
    const data = resp?.data?.data || {}
    const available = Array.isArray(data.available_databases) ? data.available_databases : []
    const defaults = Array.isArray(data.default_databases) && data.default_databases.length ? data.default_databases : ['pubchem']

    availableDatabases.value = available
    defaultDatabases.value = defaults

    if (!Array.isArray(chooseForm.databases) || !chooseForm.databases.length) {
      chooseForm.databases = [...defaults]
    } else {
      chooseForm.databases = chooseForm.databases.filter((db) => available.includes(db))
      if (!chooseForm.databases.length) chooseForm.databases = [...defaults]
    }
  } catch (error) {
    availableDatabases.value = ['pubchem']
    defaultDatabases.value = ['pubchem']
    if (!chooseForm.databases?.length) chooseForm.databases = ['pubchem']
  }
}

onMounted(() => {
  const savedUpload = getSessionByUser(sessionKeys.upload, getSessionUsername())
  if (savedUpload?.files) {
    paths.taskId = savedUpload.task_id || savedUpload.taskId || ''
    paths.statasPath = savedUpload.files.statasPath || savedUpload.files.statas_path || ''
    paths.fragtreesPath = savedUpload.files.fragtreesPath || savedUpload.files.fragtrees_path || ''
    paths.spectraPath = savedUpload.files.spectraPath || savedUpload.files.spectra_path || ''
    batchSummary.value = savedUpload.batch_summary || null
    uploadState.value = paths.taskId ? 'finished' : 'idle'
  }

  const savedChoose = getSessionByUser(sessionKeys.choose, getSessionUsername())
  if (savedChoose?.data) chooseData.value = savedChoose.data

  const savedRetrieve = getSessionByUser(sessionKeys.retrieve, getSessionUsername())
  if (savedRetrieve) {
    retrieveSummary.value = savedRetrieve?.result || null
    retrieveJob.value = {
      job_id: savedRetrieve?.job_id || '',
      status: savedRetrieve?.status || '',
      error: savedRetrieve?.error || null,
      failed_count: Number(savedRetrieve?.failed_count || 0),
      total_count: Number(savedRetrieve?.total_count || 0),
    }
  }

  loadCandidateDatabases()
})

onBeforeUnmount(() => {
  stopRetrievePolling = true
  stopUploadPolling = true
  if (resultPlotRef.value) Plotly.purge(resultPlotRef.value)
})

const onMgfChange = (_file, fileList) => {
  uploadForm.mgfFiles = (fileList || []).map((item) => item.raw).filter(Boolean)
}

const onJsonChange = (_file, fileList) => {
  uploadForm.jsonFiles = (fileList || []).map((item) => item.raw).filter(Boolean)
}

const onCustomChange = (file) => {
  chooseForm.customFile = file?.raw || null
}

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

const pollUploadResult = async (taskId) => {
  const startedAt = Date.now()
  while (!stopUploadPolling) {
    if (Date.now() - startedAt > UPLOAD_POLL_TIMEOUT_MS) throw new Error('Upload task timed out, please retry later')

    const statusResp = await getUploadFilesStatus(taskId)
    const statusData = statusResp?.data?.data
    if (!statusData) throw new Error('Failed to read upload task status')

    const progress = statusData.progress || {}
    if (progress.total) uploadProgress.value = Math.round(((progress.done || 0) / progress.total) * 100)

    if (['success', 'partial_failed'].includes(statusData.status)) {
      const files = statusData.files || {}
      paths.statasPath = files.statas_path || ''
      paths.fragtreesPath = files.fragtrees_path || ''
      paths.spectraPath = files.spectra_path || ''
      batchSummary.value = statusData.batch_summary || null
      if (statusData.statas) await mapStatasToList(statusData.statas)

      setSessionByUser(sessionKeys.upload, { task_id: paths.taskId, files: { ...paths }, batch_summary: batchSummary.value }, getSessionUsername())
      uploadState.value = 'finished'
      return statusData
    }

    if (statusData.status === 'failed') {
      const err = Array.isArray(statusData.errors) && statusData.errors.length ? statusData.errors[0]?.error : 'Sirius batch failed'
      throw new Error(err)
    }

    uploadState.value = 'processing'
    await sleep(UPLOAD_POLL_INTERVAL_MS)
  }
}

const handleUploadSubmit = async () => {
  clearGlobalError()
  if (loading.upload) return
  if (uploadState.value === 'finished') {
    ElMessage.warning('Upload already finished, please click Clear to start a new task')
    return
  }
  if (!uploadForm.mgfFiles.length) return setGlobalError('Please upload at least one mgf/txt file')

  const totalCount = uploadForm.mgfFiles.length + uploadForm.jsonFiles.length
  if (totalCount > 50) return setGlobalError('Total upload files cannot exceed 50')

  const limit = 10 * 1024 * 1024
  const hasOversize = [...uploadForm.mgfFiles, ...uploadForm.jsonFiles].some((item) => item.size > limit)
  if (hasOversize) return setGlobalError('Single file exceeds 10MB')

  const formData = new FormData()
  uploadForm.mgfFiles.forEach((file) => formData.append('files_mgf', file))
  uploadForm.jsonFiles.forEach((file) => formData.append('files_json', file))

  uploadProgress.value = 0
  uploadState.value = 'submitting'
  loading.upload = true
  stopUploadPolling = false

  try {
    const resp = await uploadFiles(formData, (evt) => {
      if (evt.total) uploadProgress.value = Math.round((evt.loaded / evt.total) * 100)
    })
    const data = resp?.data?.data
    if (!data?.task_id) throw new Error('Upload response missing task_id')

    paths.taskId = data.task_id
    chooseData.value = null
    retrieveSummary.value = null
    retrieveJob.value = null
    removeSessionByUser(sessionKeys.choose, getSessionUsername())
    removeSessionByUser(sessionKeys.retrieve, getSessionUsername())

    const isAsyncMgfOnly = data.upload_mode === 'mgf_only_async' || data.status === 'processing'
    if (isAsyncMgfOnly) {
      paths.statasPath = ''
      paths.fragtreesPath = ''
      paths.spectraPath = ''
      batchSummary.value = data.batch_summary || null
      setSessionByUser(sessionKeys.upload, { task_id: paths.taskId, files: { ...paths }, batch_summary: batchSummary.value }, getSessionUsername())

      uploadState.value = 'processing'
      ElMessage.success('MGF uploaded, Sirius queue started')
      const finalStatus = await pollUploadResult(paths.taskId)
      if (finalStatus?.status === 'partial_failed') ElMessage.warning('Upload completed with partial failures')
      else ElMessage.success('Batch upload completed')
    } else {
      if (!data?.files) throw new Error('Upload response missing output files')
      paths.statasPath = data.files.statas_path
      paths.fragtreesPath = data.files.fragtrees_path
      paths.spectraPath = data.files.spectra_path
      batchSummary.value = data.batch_summary || null

      setSessionByUser(sessionKeys.upload, { task_id: paths.taskId, files: { ...paths }, batch_summary: batchSummary.value }, getSessionUsername())
      if (data.statas) await mapStatasToList(data.statas)
      uploadState.value = 'finished'
      ElMessage.success('Batch upload completed')
    }

    currentStep.value = STEP_FILES
  } catch (error) {
    uploadState.value = 'idle'
    setGlobalError(error?.response?.data?.message || error?.message || 'Upload failed')
  } finally {
    loading.upload = false
  }
}

const handleClearUpload = () => {
  clearGlobalError()
  stopUploadPolling = true
  uploadForm.mgfFiles = []
  uploadForm.jsonFiles = []
  chooseForm.searchType = 'pubchem'
  chooseForm.ionMode = 'pos'
  chooseForm.ppmRange = 10
  chooseForm.databases = [...defaultDatabases.value]
  chooseForm.customFile = null

  paths.taskId = ''
  paths.statasPath = ''
  paths.fragtreesPath = ''
  paths.spectraPath = ''

  chooseData.value = null
  retrieveSummary.value = null
  retrieveJob.value = null
  statasData.value = null
  batchSummary.value = null
  spectraList.value = []
  fileCards.value = []
  collapseActiveFile.value = ''
  resultActiveFile.value = ''
  resultActiveTitle.value = ''
  resultPlotLoading.value = false
  resultPlotError.value = ''
  resultPlotKey.value = ''
  if (resultPlotRef.value) Plotly.purge(resultPlotRef.value)
  currentPage.value = 1
  resultCurrentPage.value = 1

  uploadProgress.value = 0
  uploadState.value = 'idle'
  currentStep.value = STEP_UPLOAD

  mgfUploadRef.value?.clearFiles()
  jsonUploadRef.value?.clearFiles()
  customUploadRef.value?.clearFiles()

  removeSessionByUser(sessionKeys.upload, getSessionUsername())
  removeSessionByUser(sessionKeys.choose, getSessionUsername())
  removeSessionByUser(sessionKeys.retrieve, getSessionUsername())

  ElMessage.success('Cleared. You can submit a new upload now')
}

const handleCandidatesSubmit = async () => {
  clearGlobalError()
  if (!paths.taskId) return setGlobalError('Please upload files first')
  if (!['pubchem', 'custom'].includes(chooseForm.searchType)) return setGlobalError('Candidate pool type is invalid')
  if (!['pos', 'neg'].includes(chooseForm.ionMode)) return setGlobalError('Ion mode is invalid')

  if (chooseForm.searchType === 'pubchem') {
    if (!chooseForm.ppmRange || Number(chooseForm.ppmRange) <= 0) return setGlobalError('PPM range must be greater than 0')
    if (!Array.isArray(chooseForm.databases) || chooseForm.databases.length === 0) return setGlobalError('Please select at least one database')
  }

  if (chooseForm.searchType === 'custom') {
    if (!chooseForm.customFile) return setGlobalError('Please upload a custom library txt file')
    const ext = (chooseForm.customFile.name || '').toLowerCase()
    if (!ext.endsWith('.txt')) return setGlobalError('Custom library only supports txt files')
  }

  const formData = new FormData()
  formData.append('search_type', chooseForm.searchType)
  formData.append('ion_mode', chooseForm.ionMode)
  if (chooseForm.searchType === 'pubchem') {
    formData.append('ppm_range', chooseForm.ppmRange)
    formData.append('databases', JSON.stringify(chooseForm.databases || []))
  }
  if (chooseForm.searchType === 'custom') formData.append('custom_lib_file', chooseForm.customFile)
  formData.append('task_id', paths.taskId)

  loading.choose = true
  try {
    const resp = await chooseCandidates(formData)
    chooseData.value = resp?.data?.data
    if (!chooseData.value) throw new Error('Candidate pool response is empty')
    if (Array.isArray(chooseData.value.available_databases)) availableDatabases.value = chooseData.value.available_databases
    if (Array.isArray(chooseData.value.databases) && chooseData.value.databases.length) chooseForm.databases = [...chooseData.value.databases]
    setSessionByUser(sessionKeys.choose, { data: chooseData.value }, getSessionUsername())
    ElMessage.success('Candidate pool selection completed')
  } catch (error) {
    setGlobalError(error?.response?.data?.message || error?.message || 'Candidate pool selection failed')
  } finally {
    loading.choose = false
  }
}

const refreshStatas = async () => {
  if (!paths.taskId && !paths.statasPath) return
  loading.statas = true
  try {
    const resp = await fetchStatas({ taskId: paths.taskId, path: paths.statasPath })
    const data = resp?.data?.data
    if (data) await mapStatasToList(data)
  } catch (error) {
    setGlobalError(error?.response?.data?.message || error?.message || 'Failed to read statas')
  } finally {
    loading.statas = false
  }
}

const pollRetrieveResult = async (jobId) => {
  const startedAt = Date.now()
  while (!stopRetrievePolling) {
    if (Date.now() - startedAt > RETRIEVE_POLL_TIMEOUT_MS) throw new Error('Search task timed out, please retry later')

    const statusResp = await getRetrieveStatus(jobId)
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
        job_id: statusData.job_id,
        status: statusData.status,
        result: retrieveSummary.value,
        error: null,
        failed_count: Number(statusData.failed_count || 0),
        total_count: Number(statusData.total_count || 0),
      }, getSessionUsername())
      await saveHistory({ normal_status: statusData.status === 'partial_failed' ? 'failed' : 'success' })
      return
    }

    if (statusData.status === 'failed') {
      const failedMsg = statusData.error || 'Search task failed'
      setSessionByUser(sessionKeys.retrieve, { job_id: statusData.job_id, status: statusData.status, result: null, error: failedMsg }, getSessionUsername())
      await saveHistory({ normal_status: 'failed' })
      throw new Error(failedMsg)
    }

    await sleep(RETRIEVE_POLL_INTERVAL_MS)
  }
}

const handleRetrieve = async () => {
  clearGlobalError()
  if (!chooseData.value) return setGlobalError('Please complete candidate pool selection first')

  stopRetrievePolling = false
  loading.retrieve = true
  retrieveSummary.value = null

  try {
    const resp = await runRetrieve(chooseData.value)
    const submitData = resp?.data?.data
    const jobId = submitData?.job_id
    if (!jobId) throw new Error('Search submission failed: missing job_id')

    retrieveJob.value = { job_id: jobId, status: submitData?.status || 'running', error: null }
    setSessionByUser(sessionKeys.retrieve, { job_id: jobId, status: retrieveJob.value.status, result: null, error: null, failed_count: 0, total_count: 0 }, getSessionUsername())

    ElMessage.success('Search task submitted and running')
    await pollRetrieveResult(jobId)
    await refreshStatas()
    currentStep.value = STEP_RESULTS
    ElMessage.success('Search completed')
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
    const resp = await downloadTaskFile({ taskId: paths.taskId, filename: 'statas.json' })
    const blob = new Blob([resp.data], { type: 'application/json;charset=utf-8' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `statas_${paths.taskId}.json`
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

    <div class="flow-nav-wrap">
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
      <section v-show="currentStep === STEP_UPLOAD" class="module-section">
        <h3 class="module-title">HEALTH CHECK</h3>
        <div class="module-body">
          <el-button class="btn-unified" :loading="loading.health" @click="handleHealthCheck">Check backend status</el-button>
        </div>

        <el-divider class="module-divider" />

        <h3 class="module-title">FILE UPLOAD</h3>
        <div class="module-body">
          <div class="upload-row">
            <div class="upload-block">
              <p class="label">mgf/txt file</p>
              <el-upload ref="mgfUploadRef" class="uploader" drag action="#" :auto-upload="false" :show-file-list="true" :limit="50" multiple :on-change="onMgfChange" accept=".mgf,.txt">
                <div class="el-upload__text">Drag or click to select mgf/txt (batch)</div>
              </el-upload>
            </div>
            <div class="upload-block">
              <p class="upload-label">json file</p>
              <el-upload ref="jsonUploadRef" class="uploader" drag action="#" :auto-upload="false" :show-file-list="true" :limit="50" multiple :on-change="onJsonChange" accept=".json">
                <div class="el-upload__text">Drag or click to select json (batch)</div>
              </el-upload>
            </div>
          </div>
          <div class="actions">
            <el-button class="btn-unified" :loading="loading.upload" :disabled="!canSubmitUpload" @click="handleUploadSubmit">{{ uploadButtonText }}</el-button>
            <el-button type="danger" plain :disabled="!canClearUpload || loading.upload" @click="handleClearUpload">Clear</el-button>
            <el-progress v-if="loading.upload" :percentage="uploadProgress" style="width: 200px" />
          </div>
          <div v-if="paths.taskId" class="hint">
            Task ID: <strong>{{ paths.taskId }}</strong>
            <el-button size="small" text @click="copyTaskId">Copy</el-button>
          </div>
          <div v-if="uploadState === 'processing'" class="hint">Sirius queue is generating json files... {{ uploadProgress }}%</div>
          <div v-if="batchSummary" class="hint">
            Paired: <strong>{{ batchSummary.paired_count || 0 }}</strong>
            <span style="margin-left: 8px">Unmatched mgf: {{ (batchSummary.unmatched_mgf || []).length }}</span>
            <span style="margin-left: 8px">Unmatched json: {{ (batchSummary.unmatched_json || []).length }}</span>
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
                <el-table-column prop="mz" label="Pepmass" width="140" />
                <el-table-column prop="adduct" label="Adduct" min-width="140" />
                <el-table-column prop="peaks" label="Peaks" width="100" />
              </el-table>
            </el-collapse-item>
          </el-collapse>
          <div v-if="fileCount > pageSize" class="pagination">
            <el-pagination background layout="prev, pager, next" :total="fileCount" :page-size="pageSize" v-model:current-page="currentPage" />
          </div>
        </div>
      </section>

      <section v-show="currentStep === STEP_CANDIDATES" class="module-section">
        <h3 class="module-title">CANDIDATE POOL</h3>
        <div class="module-body">
          <div class="form-row">
            <el-select v-model="chooseForm.searchType" placeholder="Select pool type" style="width: 220px">
              <el-option label="Database Search" value="pubchem" />
              <el-option label="Custom library" value="custom" />
            </el-select>
            <el-select v-model="chooseForm.ionMode" placeholder="Select ion mode" style="width: 180px">
              <el-option label="Positive (pos)" value="pos" />
              <el-option label="Negative (neg)" value="neg" />
            </el-select>
            <el-input-number v-if="chooseForm.searchType === 'pubchem'" v-model="chooseForm.ppmRange" :min="0" :step="1" placeholder="ppm range (>0)" style="width: 200px" />
            <el-select
              v-if="chooseForm.searchType === 'pubchem'"
              v-model="chooseForm.databases"
              multiple
              collapse-tags
              collapse-tags-tooltip
              placeholder="Select databases"
              style="width: 320px"
            >
              <el-option v-for="db in availableDatabases" :key="db" :label="db" :value="db" />
            </el-select>
            <el-upload v-if="chooseForm.searchType === 'custom'" ref="customUploadRef" class="uploader" action="#" :auto-upload="false" :show-file-list="true" :limit="1" :on-change="onCustomChange" accept=".txt">
              <el-button>Select custom library (txt)</el-button>
            </el-upload>
          </div>
          <div class="actions">
            <el-button class="btn-unified" :loading="loading.choose" @click="handleCandidatesSubmit">Confirm pool</el-button>
            <el-button class="btn-unified" :loading="loading.retrieve" @click="handleRetrieve">Run search</el-button>
            <el-button type="warning" :disabled="!canDownloadStatas" @click="handleDownloadStatas">Export results</el-button>
          </div>
          <div v-if="retrieveJob" class="hint">
            Status: {{ retrieveJob.status }}
            <span v-if="retrieveJob.status === 'partial_failed'">| Failed/Total: {{ retrieveJob.failed_count || 0 }}/{{ retrieveJob.total_count || 0 }}</span>
            <span v-if="retrieveJob.error">| Error: {{ retrieveJob.error }}</span>
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
                      @click="handleSpectrumSelect(row.title)"
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
                  <div class="normal-results-title">Normal search Top100 details</div>
                  <el-empty v-if="!normalResultEntries.length" description="No normal search results" />
                  <el-table v-else :data="normalResultEntries" size="small" class="result-table" :header-cell-style="{ background: '#f8fafc', fontWeight: '600' }">
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
  grid-template-columns: repeat(4, 1fr);
  gap: 4px;
  position: sticky;
  top: 100px;
  z-index: 900;
}
.flow-line {
  position: absolute;
  left: 12.5%;
  right: 12.5%;
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

.upload-row { display: flex; gap: 16px; flex-wrap: wrap; }
.upload-block { flex: 1; min-width: 240px; }
.upload-label { margin: 0 0 8px; color: #606266; font-size: 12px; }
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
