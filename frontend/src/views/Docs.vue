<script setup>
import { onMounted, ref } from 'vue'
import { fetchDocsContent } from '../api/docs'

const loading = ref(false)
const items = ref([])
const errorText = ref('')

const loadDocs = async () => {
  loading.value = true
  errorText.value = ''
  try {
    const resp = await fetchDocsContent()
    items.value = resp?.data?.data || []
  } catch (error) {
    errorText.value = error?.response?.data?.message || error?.message || '说明文档加载失败'
    items.value = []
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadDocs()
})
</script>

<template>
  <div class="docs">
    <div v-loading="loading" class="docs-body">
      <p v-if="errorText" class="error">{{ errorText }}</p>
      <template v-else>
        <template v-for="(item, idx) in items" :key="`${idx}-${item.type}`">
          <hr v-if="item.type === 'title' && idx > 0" class="divider" />
          <h1 v-if="item.type === 'title'" class="title">{{ item.content }}</h1>
          <h2 v-else-if="item.type === 'subtitle'" class="subtitle">{{ item.content }}</h2>
          <p v-else class="text">{{ item.content }}</p>
        </template>
      </template>
    </div>
  </div>
</template>

<style scoped>
.docs {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
  padding: 20px 24px;
  color: #1f2937;
}

.docs-body {
  min-height: 120px;
}

.divider {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 22px 0;
}

.title {
  margin: 16px 0 12px;
  font-size: 30px;
  line-height: 1.35;
  font-weight: 800;
  color: #0f172a;
}

.subtitle {
  margin: 14px 0 8px;
  font-size: 22px;
  line-height: 1.45;
  font-weight: 700;
  color: #111827;
}

.text {
  margin: 6px 0;
  font-size: 15px;
  line-height: 1.9;
  color: #374151;
  white-space: pre-wrap;
  word-break: break-word;
}

.error {
  margin: 0;
  color: #ef4444;
}
</style>
