import { useEffect, useState } from 'react'

function fmtTime(s) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}

export function CallScreen({ session }) {
  const { status, clip, elapsedS, rollingHumanConfidence, chunks, error, chunkSeconds } = session
  const [ringingDots, setRingingDots] = useState('')

  useEffect(() => {
    if (status !== 'ringing') return
    const t = setInterval(() => setRingingDots((d) => (d.length >= 3 ? '' : d + '.')), 400)
    return () => clearInterval(t)
  }, [status])

  if (status === 'ringing') {
    return (
      <div className="call-screen incoming">
        <div className="avatar pulse">📞</div>
        <h2>{clip ? clip.id : 'Unknown Caller'}</h2>
        <p className="sub">Incoming call{ringingDots}</p>
        <div className="call-actions">
          <button className="btn decline" onClick={session.decline}>
            Decline
          </button>
          <button className="btn accept" onClick={session.accept}>
            Accept
          </button>
        </div>
      </div>
    )
  }

  const humanPct =
    rollingHumanConfidence == null ? null : Math.round(rollingHumanConfidence * 100)
  const verdictText = humanPct == null ? 'Analyzing…' : `${humanPct}% likely human`

  return (
    <div className="call-screen active">
      <div className="avatar small">📞</div>
      <h2>{clip ? clip.id : 'Unknown Caller'}</h2>
      <p className="timer">{fmtTime(elapsedS)}</p>

      <div className="confidence-box">
        <p className="confidence-label">Voice authenticity</p>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{
              width: `${humanPct ?? 0}%`,
              background:
                humanPct == null ? '#888' : humanPct >= 50 ? '#22c55e' : '#ef4444',
            }}
          />
        </div>
        <p className={`confidence-text ${humanPct != null && humanPct < 50 ? 'danger' : ''}`}>
          {verdictText}
        </p>
      </div>

      {error && <p className="error">{error}</p>}

      <p className="chunk-note">
        Analyzed {chunks.length} chunk{chunks.length === 1 ? '' : 's'} ({chunkSeconds}s windows)
      </p>

      <button className="btn decline big" onClick={session.endCall}>
        End Call
      </button>
    </div>
  )
}
