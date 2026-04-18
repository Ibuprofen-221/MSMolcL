import http from './http'

export function fetchSpectrumPlot({ taskId, title } = {}) {
  return http.get('/api/spectrum/plot', {
    params: {
      task_id: taskId,
      title,
      _t: Date.now(),
    },
  })
}
