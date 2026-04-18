import http from './http'

export function fetchDocsContent() {
  return http.get('/api/docs-content')
}
