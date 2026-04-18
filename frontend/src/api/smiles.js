import http from './http'

export function visualizeSmiles(smilesList = []) {
  return http.post('/api/smiles/visualize', {
    smiles_list: smilesList,
  })
}

export function toImageUrl(imageUrl = '') {
  if (!imageUrl) return ''
  if (imageUrl.startsWith('http://') || imageUrl.startsWith('https://')) return imageUrl
  return `${http.defaults.baseURL}${imageUrl}`
}
