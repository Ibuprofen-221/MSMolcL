import http from './http'

export function downloadTaskFile({ taskId, filename, taskSpace = 'normal' }) {
  return http.get('/api/download-file', {
    params: {
      task_id: taskId,
      filename,
      task_space: taskSpace,
    },
    responseType: 'blob',
  })
}
