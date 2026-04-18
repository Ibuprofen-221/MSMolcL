import http from './http'

export function runRetrieve(payload) {
  return http.post('/api/retrieve', payload)
}

export function getRetrieveStatus(jobId) {
  return http.get('/api/retrieve/status', {
    params: { job_id: jobId },
  })
}

export function runRetrieveAdvanced(payload) {
  return http.post('/api/retrieve-advanced', payload)
}

export function getRetrieveAdvancedStatus(jobId) {
  return http.get('/api/retrieve-advanced/status', {
    params: { job_id: jobId },
  })
}
