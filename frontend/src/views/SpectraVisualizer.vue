<template>
  <section class="spectra-explorer">
    <el-card class="page-header" shadow="never">
      <template #header>
        <div class="header-row">
          <h2>谱图可视化（测试）</h2>
          <span class="subtitle">数据源：`/public/ms2_output.json`</span>
        </div>
      </template>
      <p class="desc"></p>
    </el-card>

    <div v-if="loading" class="status loading">谱图加载中...</div>
    <div v-else-if="error" class="status error">加载失败：{{ error }}</div>

    <div v-else class="spectra-list">
      <el-card
        v-for="(item, index) in spectraData"
        :key="item.title || index"
        class="spectrum-card"
        shadow="hover"
      >
        <div class="meta-info">
          <span class="title">#{{ index + 1 }} {{ item.title || 'Untitled' }}</span>
          <span class="mz">Precursor m/z: {{ item.precursor_mz ?? '-' }}</span>
        </div>
        <div :id="`plotly-canvas-${index}`" class="canvas-container"></div>
      </el-card>
    </div>
  </section>
</template>

<script setup>
import { nextTick, onMounted, onUnmounted, ref } from 'vue'
import Plotly from 'plotly.js-dist-min'

const spectraData = ref([])
const loading = ref(true)
const error = ref('')

const renderSinglePlot = (item, index) => {
  const element = document.getElementById(`plotly-canvas-${index}`)
  if (!element || !item?.plotly_data) return

  const { data, layout } = item.plotly_data
  if (!Array.isArray(data) || !layout) return

  Plotly.newPlot(
    element,
    data,
    layout,
    {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: `spectrum_${item.title || index + 1}`,
        height: 500,
        width: 1200,
        scale: 2,
      },
    }
  )
}

const initSpectra = async () => {
  loading.value = true
  error.value = ''

  try {
    const response = await fetch('/ms2_output.json')
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const payload = await response.json()
    if (!Array.isArray(payload)) {
      throw new Error('数据格式错误：期望数组')
    }

    spectraData.value = payload

    loading.value = false

    await nextTick()
    spectraData.value.forEach((item, index) => renderSinglePlot(item, index))
  } catch (err) {
    error.value = err?.message || '未知错误'
    console.error('Plotly Init Error:', err)
  } 
    
  
}

onMounted(() => {
  initSpectra()
})

onUnmounted(() => {
  spectraData.value.forEach((_, index) => {
    const element = document.getElementById(`plotly-canvas-${index}`)
    if (element) {
      Plotly.purge(element)
    }
  })
})
</script>

<style scoped>
.spectra-explorer {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.page-header {
  border: 1px solid #e5e7eb;
}

.header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

h2 {
  margin: 0;
  font-size: 22px;
  color: #1f2937;
}

.subtitle {
  color: #4b5563;
  font-size: 13px;
}

.desc {
  margin: 0;
  color: #374151;
}

.status {
  border-radius: 8px;
  padding: 12px 14px;
  font-size: 14px;
}

.status.loading {
  background: #ecf5ff;
  color: #1d4ed8;
}

.status.error {
  background: #fef2f2;
  color: #b91c1c;
}

.spectra-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.spectrum-card {
  border: 1px solid #e5e7eb;
}

.meta-info {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
  font-weight: 600;
  color: #1f2937;
}

.canvas-container {
  width: 100%;
  min-height: 450px;
}
</style>
