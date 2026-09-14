function TrendChart({ chunks, chunkSeconds }) {
  if (chunks.length === 0) return null
  const W = 460
  const H = 140
  const pad = 24
  // per-chunk P(human): y = 0% human at bottom, 100% human at top
  const pts = chunks.map((c, i) => ({
    x: pad + (i * (W - 2 * pad)) / Math.max(chunks.length - 1, 1),
    y: H - pad - (H - 2 * pad) * (c.is_synthetic ? 1 - c.confidence : c.confidence),
    human: !c.is_synthetic,
    conf: c.confidence,
    i,
  }))
  const path = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  return (
    <svg width={W} height={H} className="trend">
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#444" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#444" />
      {/* 50% threshold */}
      <line
        x1={pad}
        y1={pad + (H - 2 * pad) * 0.5}
        x2={W - pad}
        y2={pad + (H - 2 * pad) * 0.5}
        stroke="#666"
        strokeDasharray="4 4"
      />
      <path d={path} fill="none" stroke="#60a5fa" strokeWidth={2} />
      {pts.map((p) => (
        <circle key={p.i} cx={p.x} cy={p.y} r={4} fill={p.human ? '#22c55e' : '#ef4444'} />
      ))}
      <text x={pad - 14} y={pad + 4} fill="#999" fontSize={10}>
        100% human
      </text>
      <text x={pad - 14} y={H - pad + 4} fill="#999" fontSize={10}>
        0%
      </text>
    </svg>
  )
}

export function VerdictScreen({ session, onBack }) {
  const { clip, chunks, rollingHumanConfidence, sessionId, chunkSeconds } = session
  const humanPct =
    rollingHumanConfidence == null ? null : Math.round(rollingHumanConfidence * 100)
  const isLikelyHuman = humanPct != null && humanPct >= 50

  return (
    <div className="verdict-screen">
      <h2>Call Analysis</h2>
      <p className="sub">
        Clip: <strong>{clip ? clip.id : '—'}</strong> · Session{' '}
        <code>{sessionId || '—'}</code>
      </p>

      <div className={`verdict-box ${isLikelyHuman ? 'human' : 'fake'}`}>
        <p className="verdict-title">
          {humanPct == null
            ? 'Not enough data'
            : isLikelyHuman
              ? '✅ Voice likely HUMAN'
              : '⚠️ Voice likely AI-GENERATED'}
        </p>
        <p className="verdict-sub">
          Rolling voice authenticity: {humanPct == null ? '—' : `${humanPct}%`}
        </p>
      </div>

      <h3>Per-chunk confidence trend</h3>
      <TrendChart chunks={chunks} chunkSeconds={chunkSeconds} />

      <table className="chunk-table">
        <thead>
          <tr>
            <th>Chunk</th>
            <th>Window</th>
            <th>Verdict</th>
            <th>Confidence</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {chunks.map((c) => (
            <tr key={c.chunk_id}>
              <td>{c.chunk_id.split('_chunk')[1]}</td>
              <td>
                {c.offset_s}s–{c.offset_s + chunkSeconds}s
              </td>
              <td className={c.is_synthetic ? 'danger' : 'ok'}>
                {c.is_synthetic ? 'synthetic' : 'human'}
              </td>
              <td>{(c.confidence * 100).toFixed(1)}%</td>
              <td>{c.latency_ms}ms</td>
            </tr>
          ))}
        </tbody>
      </table>

      <button className="btn accept big" onClick={onBack}>
        New Call
      </button>
    </div>
  )
}
