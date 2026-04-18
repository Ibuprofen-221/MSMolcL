import { createRouter, createWebHistory } from 'vue-router'
import AdvancedSearch from '../components/AdvancedSearch.vue'
import ChemSearch from '../components/ChemSearch.vue'
import { getToken } from '../utils/storage'
import Docs from '../views/Docs.vue'
import HistoryRecords from '../views/HistoryRecords.vue'
import Login from '../views/Login.vue'
import TaskDetail from '../views/TaskDetail.vue'

const routes = [
  { path: '/', redirect: '/search' },
  { path: '/login', name: 'Login', component: Login, meta: { public: true, plainLayout: true } },
  { path: '/search', name: 'Search', component: ChemSearch, meta: { keepAlive: true } },
  { path: '/search-advanced', name: 'SearchAdvanced', component: AdvancedSearch, meta: { keepAlive: true } },
  { path: '/history', name: 'History', component: HistoryRecords },
  { path: '/docs', name: 'Docs', component: Docs },
  { path: '/task-detail/:taskId', name: 'TaskDetail', component: TaskDetail, meta: { plainLayout: true } },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

router.beforeEach((to) => {
  if (to.meta?.public) return true
  const token = getToken()
  if (token) return true
  return {
    path: '/login',
    query: { redirect: to.fullPath },
  }
})

router.afterEach(() => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
})

export default router
