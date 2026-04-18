import http from './http'

export function uploadFiles(formData, onUploadProgress) {
  return http.post('/api/upload-files', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  })
}

