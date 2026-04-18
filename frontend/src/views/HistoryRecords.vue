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

const syncNoteDrafts = (rows) => {
  const nextDrafts = {}
  for (const row of rows) {
    const taskId = row?.task_id
    if (!taskId) continue
    nextDrafts[taskId] = row?.note || ''
  }
  noteDrafts.value = nextDrafts
}

const loadHistory = async () => {
  loading.records = true
  try {
    const resp = await fetchHistoryRecords()
    const rows = resp?.data?.data?.records || []
    records.value = Array.isArray(rows) ? rows : []
    syncNoteDrafts(records.value)
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
            <el-table-column prop="normal_status" label="Normal" width="120" />
            <el-table-column prop="advanced_status" label="Advanced" width="120" />
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
