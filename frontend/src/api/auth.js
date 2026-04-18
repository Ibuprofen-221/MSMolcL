import http from './http'

export function register(payload) {
  return http.post('/register', payload)
}

export function login(payload) {
  return http.post('/login', payload)
}

export function getMyData() {
  return http.get('/my-data')
}

export function putMyData(data) {
  return http.put('/my-data', { data })
}
