import http from './http'

export function fetchHistoryRecords() {
  return http.get('/api/history')
}

export function updateHistoryRecord(payload) {
  return http.post('/api/history/update', payload)
}

export function deleteHistoryRecord(taskId) {
  return http.delete(`/api/history/${taskId}`)
}
