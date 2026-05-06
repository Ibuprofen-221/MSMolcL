const DEFAULT_FILE_NAME = 'merged_task'

const toSpectraRows = (statas) => {
  const list = statas?.['碎裂树文件统计']?.['有效碎裂树根节点信息']
  if (!Array.isArray(list)) return []
  return list
    .filter((item) => item && item.title)
    .map((item) => ({
      title: item.title,
      mz: item.mz,
      adduct: item.adduct,
      peaks: Number(item.peaks || 0),
      result: item['检索结果'] || null,
    }))
}

const toSafeFileName = (item, idx) => {
  const mgfName = item?.source_files?.mgf
  const pairKey = item?.pair_key
  return String(mgfName || pairKey || `file_${idx + 1}`).trim()
}

const normalizeStatasPath = (path, resultType) => {
  const raw = String(path || '').trim()
  if (!raw) return ''
  if (resultType !== 'advanced') return raw
  if (raw.endsWith('/statas.json')) return `${raw.slice(0, -'/statas.json'.length)}/statas_advanced.json`
  if (raw.endsWith('\\statas.json')) return `${raw.slice(0, -'\\statas.json'.length)}\\statas_advanced.json`
  return raw
}

export async function buildBatchFileCards({
  taskStatas,
  fetchStatasByPath,
  resultType = 'normal',
} = {}) {
  const batchItems = Array.isArray(taskStatas?.['批次文件']) ? taskStatas['批次文件'] : []

  if (!batchItems.length) {
    return [
      {
        fileKey: 'single_file_0',
        fileName: DEFAULT_FILE_NAME,
        pairKey: '',
        statasPath: '',
        spectra: toSpectraRows(taskStatas),
      },
    ]
  }

  const jobs = batchItems.map(async (item, idx) => {
    const rawPath = item?.output_files?.statas || ''
    const statasPath = normalizeStatasPath(rawPath, resultType)
    let statas = null

    if (statasPath && typeof fetchStatasByPath === 'function') {
      const resp = await fetchStatasByPath(statasPath)
      statas = resp?.data?.data || null
    }

    return {
      fileKey: String(item?.pair_key || `file_${idx + 1}`),
      fileName: toSafeFileName(item, idx),
      pairKey: String(item?.pair_key || ''),
      statasPath,
      spectra: toSpectraRows(statas),
    }
  })

  const settled = await Promise.allSettled(jobs)

  return settled.map((row, idx) => {
    if (row.status === 'fulfilled') return row.value
    console.error('load batch statas failed', row.reason)
    const item = batchItems[idx] || {}
    return {
      fileKey: String(item?.pair_key || `file_${idx + 1}`),
      fileName: toSafeFileName(item, idx),
      pairKey: String(item?.pair_key || ''),
      statasPath: normalizeStatasPath(item?.output_files?.statas || '', resultType),
      spectra: [],
    }
  })
}

export function flattenFileCardSpectra(fileCards = []) {
  return fileCards.flatMap((item) => (Array.isArray(item?.spectra) ? item.spectra : []))
}

export function buildTitleMapFromRows(rows = []) {
  const map = {}
  rows.forEach((item) => {
    if (item?.title) {
      map[item.title] = item
    }
  })
  return map
}
