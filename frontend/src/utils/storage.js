const isBrowser = typeof window !== 'undefined'

const TOKEN_KEY = 'auth.token'
const USER_KEY = 'auth.user'

const SESSION_USER_PREFIX = 'u:'
const BUSINESS_SESSION_KEYS = [
  'uploadFilesInfo',
  'retrieveConfig',
  'retrieveSummary',
  'advanced.taskInfo',
  'advanced.retrieveConfig',
  'advanced.retrieveSummary',
]

export function setSession(key, value) {
  if (!isBrowser) return
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value))
  } catch (error) {
    console.error('sessionStorage set failed', error)
  }
}

export function getSession(key) {
  if (!isBrowser) return null
  try {
    const raw = window.sessionStorage.getItem(key)
    return raw ? JSON.parse(raw) : null
  } catch (error) {
    console.error('sessionStorage get failed', error)
    return null
  }
}

export function removeSession(key) {
  if (!isBrowser) return
  try {
    window.sessionStorage.removeItem(key)
  } catch (error) {
    console.error('sessionStorage remove failed', error)
  }
}

export function toUserSessionKey(baseKey, username) {
  const safeBase = String(baseKey || '').trim()
  if (!safeBase) return ''
  const safeUser = String(username || '').trim()
  if (!safeUser) return ''
  return `${SESSION_USER_PREFIX}${safeUser}:${safeBase}`
}

export function setSessionByUser(baseKey, value, username) {
  const scopedKey = toUserSessionKey(baseKey, username)
  if (!scopedKey) return
  setSession(scopedKey, value)
}

export function getSessionByUser(baseKey, username) {
  const scopedKey = toUserSessionKey(baseKey, username)
  if (!scopedKey) return null
  return getSession(scopedKey)
}

export function removeSessionByUser(baseKey, username) {
  const scopedKey = toUserSessionKey(baseKey, username)
  if (!scopedKey) return
  removeSession(scopedKey)
}

export function clearBusinessSession() {
  if (!isBrowser) return
  try {
    const targets = []
    const keyLength = window.sessionStorage.length
    for (let i = 0; i < keyLength; i += 1) {
      const key = window.sessionStorage.key(i)
      if (!key) continue
      if (BUSINESS_SESSION_KEYS.includes(key)) {
        targets.push(key)
        continue
      }
      if (BUSINESS_SESSION_KEYS.some((baseKey) => key.endsWith(`:${baseKey}`) && key.startsWith(SESSION_USER_PREFIX))) {
        targets.push(key)
      }
    }
    targets.forEach((key) => window.sessionStorage.removeItem(key))
  } catch (error) {
    console.error('sessionStorage business clear failed', error)
  }
}

export function setToken(token) {
  if (!isBrowser) return
  if (!token) return
  window.localStorage.setItem(TOKEN_KEY, token)
}

export function getToken() {
  if (!isBrowser) return ''
  return window.localStorage.getItem(TOKEN_KEY) || ''
}

export function removeToken() {
  if (!isBrowser) return
  window.localStorage.removeItem(TOKEN_KEY)
}

export function setCurrentUser(user) {
  if (!isBrowser) return
  if (!user) return
  window.localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function getCurrentUser() {
  if (!isBrowser) return null
  try {
    const raw = window.localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch (error) {
    console.error('localStorage get user failed', error)
    return null
  }
}

export function removeCurrentUser() {
  if (!isBrowser) return
  window.localStorage.removeItem(USER_KEY)
}

export function clearAuth() {
  removeToken()
  removeCurrentUser()
}
