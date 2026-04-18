<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import bannerImg from '../assets/banner1.png'

const emit = defineEmits(['visibility-change'])

const BANNER_HEIGHT = 200
const HIDE_SCROLL_THRESHOLD = 100
const isHidden = ref(false)

const emitVisibility = () => {
  emit('visibility-change', {
    hidden: isHidden.value,
    offset: isHidden.value ? 0 : BANNER_HEIGHT,
  })
}

const handleScroll = () => {
  const y = window.scrollY || document.documentElement.scrollTop
  const nextHidden = y > HIDE_SCROLL_THRESHOLD
  if (nextHidden !== isHidden.value) {
    isHidden.value = nextHidden
    emitVisibility()
  }
}

onMounted(() => {
  emitVisibility()
  window.addEventListener('scroll', handleScroll, { passive: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', handleScroll)
})

const bannerStyle = computed(() => ({
  backgroundImage: `url(${bannerImg})`
}))
</script>

<template>
  <div class="hero" :class="{ 'is-hidden': isHidden }" :style="bannerStyle">
    <div class="overlay">
      <div class="title">Online Chemical Library Search Tool</div>
    </div>
  </div>
</template>

<style scoped>
.hero {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 200px;
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  transition: all 0.3s ease-in-out;
  z-index: 998;
}

.is-hidden {
  opacity: 0;
  transform: translateY(-100%);
}

.overlay {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  background: transparent;
}

.title {
  font-size: 36px;
  font-weight: 700;
  color: #ffffff;
}
</style>
