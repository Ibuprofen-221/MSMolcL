import http from './http'

export function uploadFiles(formData, onUploadProgress) {
  return http.post('/api/upload-files', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  })
}

export function getUploadFilesStatus(taskId) {
  return http.get('/api/upload-files/status', {
    params: { task_id: taskId },
  })
}

