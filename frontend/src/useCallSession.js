import { useCallback, useEffect, useRef, useState } from 'react'
import { classifyChunk, startCallSession } from './api'

const CHUNK_SECONDS = 3

/**
 * Drives a simulated call: ring → accept → active (playback + chunked
 * classification every CHUNK_SECONDS) → ended.
 *
 * The frontend tracks time offsets only; the backend slices + classifies.
 * Rolling confidence = EMA over per-chunk P(human).
 */
export function useCallSession() {
  const [status, setStatus] = useState('idle') // idle | ringing | active | ended | error
  const [clip, setClip] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [elapsedS, setElapsedS] = useState(0)
  const [chunks, setChunks] = useState([]) // {chunk_id, is_synthetic, confidence, latency_ms, offset_s}
  const [rollingHumanConfidence, setRollingHumanConfidence] = useState(null)
  const [error, setError] = useState(null)

  const audioRef = useRef(null)
  const tickRef = useRef(null)
  const nextChunkRef = useRef(0)
  const inflightRef = useRef(false)

  const clearTimers = useCallback(() => {
    if (tickRef.current) clearInterval(tickRef.current)
    tickRef.current = null
  }, [])

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
  }, [])

  const ring = useCallback((selectedClip) => {
    setError(null)
    setChunks([])
    setRollingHumanConfidence(null)
    setElapsedS(0)
    nextChunkRef.current = 0
    setClip(selectedClip)
    setStatus('ringing')
  }, [])

  const classifyUpTo = useCallback(
    async (currentSecond) => {
      if (!clip || !sessionId || inflightRef.current) return
      const targetIndex = Math.floor(currentSecond / CHUNK_SECONDS)
      if (targetIndex < nextChunkRef.current) return
      const idx = nextChunkRef.current
      const offsetS = idx * CHUNK_SECONDS
      if (offsetS >= clip.duration_s) return
      inflightRef.current = true
      try {
        const res = await classifyChunk({
          clipId: clip.id,
          sessionId,
          chunkIndex: idx,
          offsetS,
          durationS: CHUNK_SECONDS,
        })
        nextChunkRef.current = idx + 1
        setChunks((prev) => [...prev, { ...res, offset_s: offsetS }])
        // Convert to P(human) for the rolling bar
        const humanProb = res.is_synthetic ? 1 - res.confidence : res.confidence
        setRollingHumanConfidence((prev) =>
          prev == null ? humanProb : 0.6 * humanProb + 0.4 * prev,
        )
      } catch (e) {
        setError(e.message)
        setStatus('error')
      } finally {
        inflightRef.current = false
      }
    },
    [clip, sessionId],
  )

  const accept = useCallback(async () => {
    try {
      const { session_id } = await startCallSession()
      setSessionId(session_id)
      setStatus('active')
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }, [])

  const endCall = useCallback(() => {
    stopAudio()
    clearTimers()
    setStatus('ended')
  }, [stopAudio, clearTimers])

  const decline = useCallback(() => {
    stopAudio()
    clearTimers()
    setStatus('idle')
  }, [stopAudio, clearTimers])

  // After 'active', start playback; when the clip ends, end the call.
  useEffect(() => {
    if (status !== 'active' || !audioRef.current || !clip) return
    const audio = audioRef.current
    audio.currentTime = 0
    audio.play().catch(() => {
      setError('Audio playback blocked by browser — click the page and retry')
    })
    const onEnded = () => endCall()
    audio.addEventListener('ended', onEnded)
    return () => audio.removeEventListener('ended', onEnded)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, clip?.id])

  // Ticker: call timer + chunk classification cadence while active
  useEffect(() => {
    if (status !== 'active') return
    const startedAt = Date.now()
    classifyUpTo(0)
    tickRef.current = setInterval(() => {
      const sec = (Date.now() - startedAt) / 1000
      setElapsedS(sec)
      classifyUpTo(sec)
    }, 250)
    return clearTimers
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, sessionId, clip?.id])

  return {
    status, clip, sessionId, elapsedS, chunks, rollingHumanConfidence, error,
    ring, accept, endCall, decline, audioRef, chunkSeconds: CHUNK_SECONDS,
  }
}
