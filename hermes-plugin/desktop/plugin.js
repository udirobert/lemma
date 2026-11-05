// lemma Audit Room — the evidence trail, inside Hermes.
// Pane: audited papers -> per-claim verdicts -> figures, served by the
// plugin_api.py backend over ctx.rest. Data URIs keep the renderer
// file-free.
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const V_COLOR = {
  supported: '#1ec2b5',
  falsified: '#d445f2',
  inconclusive: '#fab533',
  not_audited: '#64708a',
}
const chip = (v) =>
  jsx('span', {
    style: {
      color: V_COLOR[v] || '#64708a',
      fontWeight: 700,
      fontSize: '11px',
      fontFamily: 'monospace',
      textTransform: 'uppercase',
    },
    children: v || '—',
  })

function Detail({ ctx, name }) {
  const [d, setD] = useState(null)
  useEffect(() => {
    ctx.rest(`/papers/${name}`).then(setD).catch(() => setD({ error: true }))
  }, [name])
  if (!d) return jsx('div', { className: 'p-2 text-xs opacity-60', children: 'loading…' })
  const rows = (d.claims || []).map((c) =>
    jsxs('div', {
      className: 'flex items-baseline gap-2 py-0.5',
      children: [
        jsx('span', { className: 'font-mono text-xs opacity-50', children: c.id }),
        chip(c.verdict),
        jsx('span', {
          className: 'text-xs opacity-70 truncate',
          title: (d.claim_texts || {})[c.id],
          children: (d.claim_texts || {})[c.id] || '',
        }),
      ],
      key: c.id,
    })
  )
  const figs = (d.figures || []).map((f) =>
    jsx('img', {
      src: f.data_uri,
      title: `${f.claim}/${f.name}`,
      style: { width: '100%', borderRadius: '4px', marginTop: '4px' },
      key: f.claim + f.name,
    })
  )
  return jsxs('div', {
    className: 'flex flex-col gap-1 overflow-auto',
    children: [
      jsxs('div', {
        className: 'text-xs font-medium',
        children: [name, ' · judge: ', chip(d.judge)],
      }),
      ...rows,
      ...figs,
    ],
  })
}

function AuditRoom({ ctx }) {
  const [papers, setPapers] = useState(null)
  const [runs, setRuns] = useState({})
  const [sel, setSel] = useState(null)
  useEffect(() => {
    const load = () => {
      ctx.rest('/papers').then((r) => setPapers(r.papers || [])).catch(() => setPapers([]))
      ctx.rest('/runs').then((r) =>
        setRuns(Object.fromEntries((r.runs || []).map((x) => [x.workdir, x])))
      ).catch(() => {})
    }
    load()
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [])
  if (papers === null)
    return jsx('div', { className: 'p-3 text-xs opacity-60', children: 'loading…' })
  if (!papers.length)
    return jsxs('div', {
      className: 'p-3 text-xs opacity-60',
      children: [
        'No audits yet — run ',
        jsx('code', { children: 'lemma_quick_audit' }),
        ' on an arXiv id.',
      ],
    })
  const list = papers.map((p) => {
    const run = runs[p.workdir]
    return jsxs('div', {
      children: [
        jsxs('button', {
          type: 'button',
          onClick: () => setSel(p.workdir === sel ? null : p.workdir),
          className: 'flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs hover:bg-white/5',
          children: [
            run ? jsx('span', { style: { color: '#1ec2b5' }, children: '▸ live' })
                : chip(p.judge || 'run'),
            jsx('span', { className: 'truncate', children: p.workdir }),
            (p.flags || []).length > 0 &&
              jsx('span', {
                title: (p.flags || []).join(', '),
                style: { color: '#fab533' },
                children: '⚑',
              }),
          ],
        }),
        run &&
          jsx('div', {
            className: 'pl-4 text-[10px] font-mono opacity-50 truncate',
            children: (run.trace_tail || []).join(' · ') || 'working…',
          }),
      ],
      key: p.workdir,
    })
  })
  return jsxs('div', {
    className: 'flex h-full flex-col gap-1 overflow-hidden p-2 text-sm',
    children: [
      jsx('div', { className: 'text-xs font-medium opacity-80', children: 'lemma · audit room' }),
      jsx('div', { className: 'flex flex-col overflow-auto', children: list }),
      sel && jsx(Detail, { ctx, name: sel }),
    ],
  })
}

export default {
  id: 'lemma',
  name: 'Lemma Audit Room',
  register(ctx) {
    ctx.register({
      id: 'audit-room',
      area: 'panes',
      title: 'Audit Room',
      data: { placement: 'right', width: '340px' },
      render: () => jsx(AuditRoom, { ctx }),
    })
    ctx.register({
      id: 'lemma-chip',
      area: 'statusBar.right',
      order: 140,
      render: () =>
        jsx('span', {
          className: 'px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
          children: 'lemma',
        }),
    })
  },
}
