import http from './http'

export function checkHealth() {
  return http.get('/health')
}
