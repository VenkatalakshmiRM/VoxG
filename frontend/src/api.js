const API = '/api'

async function jsonFetch(url, options) {
  const res = await fetch(url, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export function fetchClips() {
  return jsonFetch(`${API}/clips`)
}

export function startCallSession() {
  return jsonFetch(`${API}/call-session`, { method: 'POST' })
}

export function classifyChunk({ clipId, sessionId, chunkIndex, offsetS, durationS }) {
  return jsonFetch(`${API}/classify-chunk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      clip_id: clipId,
      session_id: sessionId,
      chunk_index: chunkIndex,
      offset_s: offsetS,
      duration_s: durationS,
    }),
  })
}
