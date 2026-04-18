import http from './http'

// 读取检索结果文件，优先 task_id，兼容 path
export function fetchStatas({ taskId, path, taskSpace = 'normal', resultType = 'normal' } = {}) {
  const params = { task_space: taskSpace, result_type: resultType, _t: Date.now() }
  if (taskId) params.task_id = taskId
  else if (path) params.path = path
  return http.get('/api/statas', { params })
}
