import http from './http'

export function getCandidateDatabases() {
  return http.get('/api/candidate_databases')
}

export function chooseCandidates(formData) {
  return http.post('/api/candidates_choose', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
