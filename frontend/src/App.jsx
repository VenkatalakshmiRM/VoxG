import { useEffect, useState } from 'react'
import { fetchClips } from './api'
import { useCallSession } from './useCallSession'
import { CallScreen } from './CallScreen'
import { VerdictScreen } from './VerdictScreen'

export default function App() {
  const [clips, setClips] = useState([])
  const [selected, setSelected] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const session = useCallSession()

  useEffect(() => {
    fetchClips()
      .then((res) => {
        setClips(res.clips)
        if (res.clips.length > 0) setSelected(res.clips[0])
      })
      .catch((e) => setLoadError(e.message))
  }, [])

  const startRing = (clip) => {
    if (clip) session.ring(clip)
  }

  return (
    <div className="app">
      <header>
        <h1>VoxG</h1>
        <p className="tagline">Is that voice real — or AI-cloned?</p>
      </header>

      {loadError && (
        <p className="error banner">
          Cannot reach the backend. Start it with:{' '}
          <code>cd backend &amp;&amp; uvicorn app.main:app --port 8000</code>
        </p>
      )}

      {session.status === 'idle' && (
        <section className="picker">
          <h3>Choose an incoming caller</h3>
          {clips.length === 0 && !loadError && <p>Loading clips…</p>}
          <div className="clip-grid">
            {clips.map((c) => (
              <button
                key={c.id}
                className={`clip-card ${selected?.id === c.id ? 'selected' : ''}`}
                onClick={() => setSelected(c)}
              >
                <span className="clip-icon">{c.label === 'synthetic' ? '🤖' : '👤'}</span>
                <span className="clip-name">{c.id}</span>
                <span className="clip-meta">
                  {c.duration_s.toFixed(1)}s · {c.generator_type}
                </span>
              </button>
            ))}
          </div>
          <p className="picker-note">
            (Emoji = ground truth, for team use only — hidden from judges in the live demo.)
          </p>
          <button
            className="btn accept big"
            onClick={() => startRing(selected)}
            disabled={!selected}
          >
            📲 Simulate Incoming Call
          </button>
        </section>
      )}

      {session.status !== 'idle' && session.status !== 'ended' && (
        <CallScreen session={session} />
      )}

      {session.status === 'ended' && <VerdictScreen session={session} onBack={session.decline} />}

      {/* Audio element drives simulated playback; src = selected clip */}
      {selected && <audio ref={session.audioRef} src={selected.url} preload="auto" />}
    </div>
  )
}
