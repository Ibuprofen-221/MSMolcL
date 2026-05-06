<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRouter } from 'vue-router'
import { deleteHistoryRecord, fetchHistoryRecords, updateHistoryRecord } from '../api/history'

const router = useRouter()

const loading = reactive({
  records: false,
  deletingTaskId: '',
  savingNoteTaskId: '',
})

const records = ref([])
const noteDrafts = ref({})
const statusCountMap = ref({})
const statusCountLoading = reactive({})

const syncNoteDrafts = (rows) => {
  const nextDrafts = {}
  for (const row of rows) {
    const taskId = row?.task_id
    if (!taskId) continue
    nextDrafts[taskId] = row?.note || ''
  }
  noteDrafts.value = nextDrafts
}

const countSuccessSpectra = (statas) => {
  const rows = Array.isArray(statas?.['碎裂树文件统计']?.['有效碎裂树根节点信息'])
    ? statas['碎裂树文件统计']['有效碎裂树根节点信息']
    : []
  return rows.reduce((acc, row) => {
    const result = row?.['检索结果']
    const resultRows = Array.isArray(result?.result_top100) ? result.result_top100 : []
    const top100 = Array.isArray(result?.top100) ? result.top100 : []
    return acc + (resultRows.length || top100.length ? 1 : 0)
  }, 0)
}

const getTotalFiles = (normalStatas, advancedStatas) => {
  const normalBatch = Array.isArray(normalStatas?.['批次文件']) ? normalStatas['批次文件'].length : 0
  const advancedBatch = Array.isArray(advancedStatas?.['批次文件']) ? advancedStatas['批次文件'].length : 0
  if (normalBatch || advancedBatch) return Math.max(normalBatch, advancedBatch)

  const normalRows = Array.isArray(normalStatas?.['碎裂树文件统计']?.['有效碎裂树根节点信息'])
    ? normalStatas['碎裂树文件统计']['有效碎裂树根节点信息'].length
    : 0
  const advancedRows = Array.isArray(advancedStatas?.['碎裂树文件统计']?.['有效碎裂树根节点信息'])
    ? advancedStatas['碎裂树文件统计']['有效碎裂树根节点信息'].length
    : 0
  return Math.max(normalRows, advancedRows)
}

const formatStatusCount = (status, success, total) => {
  if (Number.isFinite(total) && total > 0) return `${success}/${total}`
  if (status === 'pending') return '0/0'
  if (status === 'failed') return '0/--'
  if (status === 'success') return `${success}/--`
  return '--/--'
}

const loadStatusCount = async (row) => {
  const taskId = String(row?.task_id || '').trim()
  if (!taskId || statusCountMap.value[taskId] || statusCountLoading[taskId]) return

  statusCountLoading[taskId] = true
  try {
    const [normalResp, advancedResp] = await Promise.allSettled([
      fetchStatas({ taskId, resultType: 'normal' }),
      fetchStatas({ taskId, resultType: 'advanced' }),
    ])

    const normalStatas = normalResp.status === 'fulfilled' ? normalResp.value?.data?.data || {} : {}
    const advancedStatas = advancedResp.status === 'fulfilled' ? advancedResp.value?.data?.data || {} : {}

    const total = getTotalFiles(normalStatas, advancedStatas)
    const normalSuccess = countSuccessSpectra(normalStatas)
    const advancedSuccess = countSuccessSpectra(advancedStatas)

    statusCountMap.value[taskId] = {
      normal: formatStatusCount(row?.normal_status, normalSuccess, total),
      advanced: formatStatusCount(row?.advanced_status, advancedSuccess, total),
    }
  } catch (error) {
    statusCountMap.value[taskId] = {
      normal: formatStatusCount(row?.normal_status, 0, 0),
      advanced: formatStatusCount(row?.advanced_status, 0, 0),
    }
  } finally {
    statusCountLoading[taskId] = false
  }
}

const loadHistory = async () => {
  loading.records = true
  try {
    const resp = await fetchHistoryRecords()
    const rows = resp?.data?.data?.records || []
    records.value = Array.isArray(rows) ? rows : []
    syncNoteDrafts(records.value)
    records.value.forEach((row) => loadStatusCount(row))
  } catch (error) {
    const msg = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Failed to load history'
    ElMessage.error(msg)
  } finally {
    loading.records = false
  }
}

const saveTaskNote = async (row) => {
  const taskId = row?.task_id
  if (!taskId) return

  const draft = String(noteDrafts.value[taskId] ?? '').trim()
  const oldValue = String(row?.note ?? '').trim()

  if (draft === oldValue) return
  if (loading.savingNoteTaskId === taskId) return

  loading.savingNoteTaskId = taskId
  try {
    const resp = await updateHistoryRecord({
      task_id: taskId,
      note: draft,
    })

    const serverRecord = resp?.data?.data || {}
    const nextNote = String(serverRecord.note ?? draft)
    row.note = nextNote
    noteDrafts.value[taskId] = nextNote
  } catch (error) {
    noteDrafts.value[taskId] = String(row?.note ?? '')
    const msg = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Save note failed'
    ElMessage.error(msg)
  } finally {
    loading.savingNoteTaskId = ''
  }
}

const openTaskDetail = (taskId) => {
  const route = router.resolve({ name: 'TaskDetail', params: { taskId } })
  window.open(route.href, '_blank')
}

const deleteTaskRecord = async (taskId) => {
  try {
    await ElMessageBox.confirm(
      `确认删除任务 ${taskId} 吗？该操作不可恢复。`,
      'Delete Confirmation',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning',
      },
    )
  } catch (error) {
    return
  }

  loading.deletingTaskId = taskId
  try {
    await deleteHistoryRecord(taskId)
    ElMessage.success('Delete success')
    await loadHistory()
  } catch (error) {
    const msg = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Delete failed'
    ElMessage.error(msg)
  } finally {
    loading.deletingTaskId = ''
  }
}

onMounted(() => {
  loadHistory()
})
</script>

<template>
  <div class="page">
    <el-card class="big-card">
      <section class="module-section">
        <div class="header-row">
          <h3 class="module-title">HISTORY RECORDS</h3>
          <el-button :loading="loading.records" @click="loadHistory">Refresh</el-button>
        </div>
        <div class="module-body">
          <el-table :data="records" stripe v-loading="loading.records" style="width: 100%">
            <el-table-column prop="task_id" label="Task ID" min-width="240" />
            <el-table-column prop="create_time" label="Create Time" min-width="220" />
            <el-table-column label="Normal" width="150">
              <template #default="scope">
                <span>{{ statusCountMap[scope.row.task_id]?.normal || (statusCountLoading[scope.row.task_id] ? '计算中...' : '--/--') }}</span>
              </template>
            </el-table-column>
            <el-table-column label="Advanced" width="150">
              <template #default="scope">
                <span>{{ statusCountMap[scope.row.task_id]?.advanced || (statusCountLoading[scope.row.task_id] ? '计算中...' : '--/--') }}</span>
              </template>
            </el-table-column>
            <el-table-column label="Note" min-width="280">
              <template #default="scope">
                <el-input
                  v-model="noteDrafts[scope.row.task_id]"
                  size="small"
                  placeholder="Add your note"
                  :disabled="loading.savingNoteTaskId === scope.row.task_id"
                  @blur="saveTaskNote(scope.row)"
                  @keydown.enter.prevent="saveTaskNote(scope.row)"
                />
              </template>
            </el-table-column>
            <el-table-column label="Actions" min-width="260">
              <template #default="scope">
                <div class="actions-cell">
                  <el-button link type="warning" @click="openTaskDetail(scope.row.task_id)"><strong>Detail</strong></el-button>
                  <el-button
                    class="delete-btn"
                    link
                    type="danger"
                    :loading="loading.deletingTaskId === scope.row.task_id"
                    @click="deleteTaskRecord(scope.row.task_id)"
                  >
                    <strong>Delete</strong>
                  </el-button>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </section>
    </el-card>
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

.module-title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: #000000;
}

.module-body {
  margin-top: 10px;
  font-size: 12px;
  font-weight: 400;
}

.header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.actions-cell {
  display: flex;
  align-items: center;
  gap: 10px;
}

.delete-btn {
  color: #f56c6c !important;
  margin-left: auto;
}
</style>
