<script setup>
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { login, register } from '../api/auth'
import { clearBusinessSession, getCurrentUser, setCurrentUser, setToken } from '../utils/storage'

const router = useRouter()
const route = useRoute()

const activeTab = ref('login')
const loading = ref(false)

const loginForm = reactive({
  username: '',
  password: '',
})

const registerForm = reactive({
  username: '',
  password: '',
  confirmPassword: '',
})

const afterLoginRedirect = () => {
  const target = route.query?.redirect
  if (typeof target === 'string' && target.startsWith('/')) {
    router.replace(target)
    return
  }
  router.replace('/search')
}

const handleLogin = async () => {
  if (!loginForm.username.trim() || !loginForm.password) {
    ElMessage.error('请输入用户名和密码')
    return
  }

  loading.value = true
  try {
    const resp = await login({
      username: loginForm.username.trim(),
      password: loginForm.password,
    })

    const data = resp?.data?.data || {}
    if (!data.access_token) {
      throw new Error('登录响应缺少 access_token')
    }

    const nextUsername = String(data.username || loginForm.username.trim()).trim()
    const previousUsername = String(getCurrentUser()?.username || '').trim()
    if (!previousUsername || previousUsername !== nextUsername) {
      clearBusinessSession()
    }

    setToken(data.access_token)
    setCurrentUser({ username: nextUsername })
    ElMessage.success('登录成功')
    afterLoginRedirect()
  } catch (error) {
    const msg = error?.response?.data?.detail || error?.response?.data?.message || error?.message || '登录失败'
    ElMessage.error(msg)
  } finally {
    loading.value = false
  }
}

const handleRegister = async () => {
  const username = registerForm.username.trim()
  if (!username || !registerForm.password) {
    ElMessage.error('请完整填写注册信息')
    return
  }
  if (registerForm.password !== registerForm.confirmPassword) {
    ElMessage.error('两次密码输入不一致')
    return
  }

  loading.value = true
  try {
    await register({ username, password: registerForm.password })
    ElMessage.success('注册成功，请登录')
    activeTab.value = 'login'
    loginForm.username = username
    loginForm.password = ''
  } catch (error) {
    const msg = error?.response?.data?.detail || error?.response?.data?.message || error?.message || '注册失败'
    ElMessage.error(msg)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <el-card class="login-card" shadow="hover">
      <h2 class="title">用户登录</h2>

      <el-tabs v-model="activeTab" class="tabs">
        <el-tab-pane label="登录" name="login">
          <el-form label-position="top">
            <el-form-item label="用户名">
              <el-input v-model="loginForm.username" placeholder="请输入用户名" clearable />
            </el-form-item>
            <el-form-item label="密码">
              <el-input v-model="loginForm.password" type="password" show-password placeholder="请输入密码" />
            </el-form-item>
            <el-button type="primary" :loading="loading" class="submit-btn" @click="handleLogin">登录</el-button>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="注册" name="register">
          <el-form label-position="top">
            <el-form-item label="用户名">
              <el-input v-model="registerForm.username" placeholder="仅支持字母数字下划线中划线" clearable />
            </el-form-item>
            <el-form-item label="密码">
              <el-input v-model="registerForm.password" type="password" show-password placeholder="至少 6 位" />
            </el-form-item>
            <el-form-item label="确认密码">
              <el-input v-model="registerForm.confirmPassword" type="password" show-password placeholder="再次输入密码" />
            </el-form-item>
            <el-button type="success" :loading="loading" class="submit-btn" @click="handleRegister">注册</el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #e7f0fb 0%, #f7fbff 100%);
}

.login-card {
  width: min(460px, 92vw);
  border-radius: 12px;
}

.title {
  margin: 4px 0 12px;
  text-align: center;
  font-size: 24px;
  color: #1f2937;
}

.submit-btn {
  width: 100%;
}
</style>
