<script setup>
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import Plotly from 'plotly.js-dist-min'
import { ElMessage } from 'element-plus'
import { fetchSpectrumPlot } from '../api/spectrum'
import { toImageUrl, visualizeSmiles } from '../api/smiles'

const props = defineProps({
  taskId: {
    type: String,
    required: true,
  },
  title: {
    type: String,
    required: true,
  },
  normalItem: {
    type: Object,
    default: () => ({}),
  },
  advancedItem: {
    type: Object,
    default: () => ({}),
  },
})

const emit = defineEmits(['close'])

const plotRef = ref(null)
const plotLoading = ref(false)
const smilesInput = ref('')
const rankResult = ref(null)

const imageState = reactive({
  normal: {},
  advanced: {},
})
const imageLoading = reactive({
  normal: false,
  advanced: false,
})
const imageLoadedKey = reactive({
  normal: '',
  advanced: '',
})

const toEntries = (result) => {
  const rows = Array.isArray(result?.result_top100) ? result.result_top100 : []
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
}

const normalEntries = computed(() => toEntries(props.normalItem?.['检索结果']))
const advancedEntries = computed(() => toEntries(props.advancedItem?.['检索结果']))


const displayAdduct = computed(() => props.normalItem?.adduct || props.advancedItem?.adduct || '-')
const displayMz = computed(() => props.normalItem?.mz ?? props.advancedItem?.mz ?? '-')

const getImageMeta = (mode, smiles) => {
  return imageState[mode]?.[smiles] || { status: 'idle', image_src: '' }
}

const loadImagesForMode = async (mode) => {
  const rows = mode === 'normal' ? normalEntries.value : advancedEntries.value
  const smilesList = rows.map((item) => item.smiles).filter(Boolean)
  const uniqueSmiles = [...new Set(smilesList)]
  const cacheKey = uniqueSmiles.join('|')

  if (!uniqueSmiles.length) {
    imageState[mode] = {}
    imageLoadedKey[mode] = ''
    return
  }
  if (imageLoadedKey[mode] === cacheKey && Object.keys(imageState[mode] || {}).length) {
    return
  }

  imageLoading[mode] = true
  try {
    const resp = await visualizeSmiles(uniqueSmiles)
    const results = resp?.data?.data?.results || []
    const map = {}
    uniqueSmiles.forEach((smiles, idx) => {
      const row = results[idx] || {}
      const status = row.status || 'failed'
      const imageUrl = row.image_url || ''
      map[smiles] = {
        status,
        image_src: status === 'ready' ? toImageUrl(imageUrl) : '',
      }
    })
    imageState[mode] = map
    imageLoadedKey[mode] = cacheKey
  } catch (error) {
    imageState[mode] = {}
    imageLoadedKey[mode] = ''
    const msg = error?.response?.data?.message || error?.message || '结构图加载失败'
    ElMessage.error(msg)
  } finally {
    imageLoading[mode] = false
  }
}

const autoLoadImages = async () => {
  await Promise.all([
    loadImagesForMode('normal'),
    loadImagesForMode('advanced'),
  ])
}

const renderPlot = async () => {
  if (!plotRef.value) return

  plotLoading.value = true
  try {
    const resp = await fetchSpectrumPlot({ taskId: props.taskId, title: props.title })
    const payload = resp?.data?.data?.plotly_data
    if (!payload?.data || !payload?.layout) {
      throw new Error('Invalid spectrum payload')
    }

    await nextTick()
    await Plotly.react(plotRef.value, payload.data, payload.layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    })
  } catch (error) {
    const msg = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Failed to load spectrum plot'
    ElMessage.error(msg)
    if (plotRef.value) {
      Plotly.purge(plotRef.value)
    }
  } finally {
    plotLoading.value = false
  }
}

const queryRank = () => {
  const target = (smilesInput.value || '').trim()
  if (!target) {
    rankResult.value = null
    return
  }

  const normalEntry = normalEntries.value.find((item) => item.smiles === target) || null
  const advancedEntry = advancedEntries.value.find((item) => item.smiles === target) || null

  rankResult.value = {
    smiles: target,
    normalEntry,
    advancedEntry,
  }
}

watch(
  () => [props.taskId, props.title],
  () => {
    rankResult.value = null
    smilesInput.value = ''
    imageState.normal = {}
    imageState.advanced = {}
    imageLoadedKey.normal = ''
    imageLoadedKey.advanced = ''
    renderPlot()
    autoLoadImages()
  }
)

onMounted(() => {
  renderPlot()
  autoLoadImages()
})

onUnmounted(() => {
  if (plotRef.value) {
    Plotly.purge(plotRef.value)
  }
})
</script>

<template>
  <el-card class="detail-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <span class="title">{{ title }}</span>
        <el-button link type="danger" @click="emit('close', title)">✕</el-button>
      </div>
    </template>

    <div class="plot-wrapper" v-loading="plotLoading">
      <div ref="plotRef" class="plot-canvas"></div>
    </div>

    <div class="meta-row">
      <div><strong>Adduct:</strong> {{ displayAdduct }}</div>
      <div><strong>Precursor m/z:</strong> {{ displayMz }}</div>
    </div>

    <div class="top-grid">
      <div class="top-col">
        <div class="top-title-row">
          <div class="top-title">Normal search Top100 details</div>
          <span v-if="imageLoading.normal" class="image-loading-hint">结构图生成中...</span>
        </div>
        <div class="result-list">
          <div v-if="normalEntries.length === 0" class="empty-block">No normal search results</div>
          <div v-for="row in normalEntries" :key="`normal-${row.rank}-${row.smiles}`" class="result-card">
            <div class="result-left">
              <img
                v-if="getImageMeta('normal', row.smiles).status === 'ready' && getImageMeta('normal', row.smiles).image_src"
                :src="getImageMeta('normal', row.smiles).image_src"
                :alt="row.smiles"
                class="mol-image"
              />
              <div v-else class="mol-placeholder">{{ getImageMeta('normal', row.smiles).status === 'failed' ? '×' : '-' }}</div>
            </div>
            <div class="result-right">
              <div><strong>Rank:</strong> {{ row.rank }} | <strong>Score:</strong> {{ row.score ? Number(row.score).toFixed(4) : '-' }}</div>
              <div><strong>SMILES:</strong> {{ row.smiles || '-' }}</div>
              <div><strong>FORMULA:</strong> {{ row.formula || '-' }}</div>
              <div><strong>GENERIC_NAME:</strong> {{ row.generic_name || '-' }}</div>
              <div><strong>DATABASE_NAME:</strong> {{ row.database_name || '-' }}</div>
              <div><strong>DATABASE_ID:</strong> {{ row.database_id || '-' }}</div>
              <div><strong>INCHI_KEY:</strong> {{ row.inchi_key || '-' }}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="top-col">
        <div class="top-title-row">
          <div class="top-title">Advanced search Top100 details</div>
          <span v-if="imageLoading.advanced" class="image-loading-hint">结构图生成中...</span>
        </div>
        <div class="result-list">
          <div v-if="advancedEntries.length === 0" class="empty-block">No advanced search results</div>
          <div v-for="row in advancedEntries" :key="`advanced-${row.rank}-${row.smiles}`" class="result-card">
            <div class="result-left">
              <img
                v-if="getImageMeta('advanced', row.smiles).status === 'ready' && getImageMeta('advanced', row.smiles).image_src"
                :src="getImageMeta('advanced', row.smiles).image_src"
                :alt="row.smiles"
                class="mol-image"
              />
              <div v-else class="mol-placeholder">{{ getImageMeta('advanced', row.smiles).status === 'failed' ? '×' : '-' }}</div>
            </div>
            <div class="result-right">
              <div><strong>Rank:</strong> {{ row.rank }} | <strong>Score:</strong> {{ row.score ? Number(row.score).toFixed(4) : '-' }}</div>
              <div><strong>SMILES:</strong> {{ row.smiles || '-' }}</div>
              <div><strong>FORMULA:</strong> {{ row.formula || '-' }}</div>
              <div><strong>GENERIC_NAME:</strong> {{ row.generic_name || '-' }}</div>
              <div><strong>DATABASE_NAME:</strong> {{ row.database_name || '-' }}</div>
              <div><strong>DATABASE_ID:</strong> {{ row.database_id || '-' }}</div>
              <div><strong>INCHI_KEY:</strong> {{ row.inchi_key || '-' }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="query-row">
      <el-input
        v-model="smilesInput"
        placeholder="Enter a SMILES to check Top100 rank"
        clearable
        @keyup.enter="queryRank"
      />
      <el-button type="primary" @click="queryRank">Check</el-button>
    </div>

    <div v-if="rankResult" class="query-result">
      <div class="query-result-title"><strong>SMILES:</strong> {{ rankResult.smiles }}</div>
      <div class="query-card-grid">
        <div class="query-card">
          <div class="query-card-header">Normal search match</div>
          <template v-if="rankResult.normalEntry">
            <div class="query-card-body">
              <img
                v-if="getImageMeta('normal', rankResult.normalEntry.smiles).status === 'ready' && getImageMeta('normal', rankResult.normalEntry.smiles).image_src"
                :src="getImageMeta('normal', rankResult.normalEntry.smiles).image_src"
                :alt="rankResult.normalEntry.smiles"
                class="query-mol-image"
              />
              <div v-else class="query-mol-placeholder">{{ getImageMeta('normal', rankResult.normalEntry.smiles).status === 'failed' ? '×' : '-' }}</div>
              <div class="query-meta">
                <div><strong>Rank:</strong> {{ rankResult.normalEntry.rank }}</div>
                <div><strong>Score:</strong> {{ rankResult.normalEntry.score ? Number(rankResult.normalEntry.score).toFixed(4) : '-' }}</div>
                <div><strong>FORMULA:</strong> {{ rankResult.normalEntry.formula || '-' }}</div>
                <div><strong>GENERIC_NAME:</strong> {{ rankResult.normalEntry.generic_name || '-' }}</div>
                <div><strong>DATABASE_NAME:</strong> {{ rankResult.normalEntry.database_name || '-' }}</div>
                <div><strong>DATABASE_ID:</strong> {{ rankResult.normalEntry.database_id || '-' }}</div>
                <div><strong>INCHI_KEY:</strong> {{ rankResult.normalEntry.inchi_key || '-' }}</div>
              </div>
            </div>
          </template>
          <div v-else class="query-empty">No matched result</div>
        </div>

        <div class="query-card">
          <div class="query-card-header">Advanced search match</div>
          <template v-if="rankResult.advancedEntry">
            <div class="query-card-body">
              <img
                v-if="getImageMeta('advanced', rankResult.advancedEntry.smiles).status === 'ready' && getImageMeta('advanced', rankResult.advancedEntry.smiles).image_src"
                :src="getImageMeta('advanced', rankResult.advancedEntry.smiles).image_src"
                :alt="rankResult.advancedEntry.smiles"
                class="query-mol-image"
              />
              <div v-else class="query-mol-placeholder">{{ getImageMeta('advanced', rankResult.advancedEntry.smiles).status === 'failed' ? '×' : '-' }}</div>
              <div class="query-meta">
                <div><strong>Rank:</strong> {{ rankResult.advancedEntry.rank }}</div>
                <div><strong>Score:</strong> {{ rankResult.advancedEntry.score ? Number(rankResult.advancedEntry.score).toFixed(4) : '-' }}</div>
                <div><strong>FORMULA:</strong> {{ rankResult.advancedEntry.formula || '-' }}</div>
                <div><strong>GENERIC_NAME:</strong> {{ rankResult.advancedEntry.generic_name || '-' }}</div>
                <div><strong>DATABASE_NAME:</strong> {{ rankResult.advancedEntry.database_name || '-' }}</div>
                <div><strong>DATABASE_ID:</strong> {{ rankResult.advancedEntry.database_id || '-' }}</div>
                <div><strong>INCHI_KEY:</strong> {{ rankResult.advancedEntry.inchi_key || '-' }}</div>
              </div>
            </div>
          </template>
          <div v-else class="query-empty">No matched result</div>
        </div>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
.detail-card {
  height: 100%;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.title {
  font-weight: 700;
  word-break: break-all;
}

.plot-wrapper {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 8px;
}

.plot-canvas {
  min-height: 360px;
}

.meta-row {
  margin-top: 12px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.top-grid {
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.top-col {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 10px;
  background: #fff;
}

.top-title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.top-title {
  font-weight: 700;
}

.result-list {
  max-height: 560px;
  overflow-y: auto;
  border-top: 1px solid #f1f5f9;
  padding-top: 8px;
}

.empty-block {
  color: #909399;
  padding: 10px 0;
}

.result-card {
  display: grid;
  grid-template-columns: 130px 1fr;
  gap: 10px;
  padding: 10px 0;
  border-bottom: 1px solid #eef2f7;
}

.result-left {
  width: 120px;
  height: 120px;
}

.mol-image {
  width: 120px;
  height: 120px;
  object-fit: contain;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  background: #f8fafc;
}

.mol-placeholder {
  width: 120px;
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed #dcdfe6;
  border-radius: 6px;
  color: #909399;
  background: #fafafa;
}

.result-right {
  font-size: 12px;
  line-height: 1.55;
  color: #303133;
  word-break: break-all;
}

.query-row {
  margin-top: 14px;
  display: flex;
  gap: 10px;
}

.query-result {
  margin-top: 10px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px;
}

.query-result-title {
  margin-bottom: 10px;
  line-height: 1.6;
}

.query-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.query-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #fff;
  overflow: hidden;
}

.query-card-header {
  padding: 8px 10px;
  font-weight: 700;
  border-bottom: 1px solid #eef2f7;
  background: #f9fbff;
}

.query-card-body {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 10px;
  padding: 10px;
}

.query-mol-image {
  width: 120px;
  height: 120px;
  object-fit: contain;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  background: #f8fafc;
}

.query-mol-placeholder {
  width: 120px;
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed #dcdfe6;
  border-radius: 6px;
  color: #909399;
  background: #fafafa;
}

.query-meta {
  font-size: 12px;
  line-height: 1.55;
  word-break: break-all;
}

.query-empty {
  padding: 10px;
  color: #909399;
}

.image-loading-hint {
  font-size: 12px;
  color: #909399;
}

@media (max-width: 1200px) {
  .top-grid,
  .meta-row,
  .query-card-grid {
    grid-template-columns: 1fr;
  }

  .result-card,
  .query-card-body {
    grid-template-columns: 1fr;
  }
}
</style>
