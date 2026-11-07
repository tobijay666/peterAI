/**
 * Inspector — click a node to see its usage score, collection records with
 * sources and confidence, an embedding preview, edges and curiosity items.
 */
import { apiFetch } from '../api/client';
import type { NodeDetail } from './types';

const esc = (s: unknown): string => String(s ?? '').replace(/[&<>"']/g, c => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));

function ago(ts: number | null): string {
  if (!ts) return '—';
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return 'just now';
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}

export interface InspectorActions {
  focus: (nodeId: string) => void;
  showRecords: (nodeId: string, collection: string) => void;
  removed: (nodeId: string) => void;
}

export class Inspector {
  readonly el: HTMLDivElement;
  private current: string | null = null;

  constructor(parent: HTMLElement, private actions: InspectorActions) {
    this.el = document.createElement('div');
    this.el.className = 'mind-inspector-panel';
    this.el.style.display = 'none';
    parent.appendChild(this.el);
  }

  get nodeId(): string | null { return this.current; }

  hide(): void { this.el.style.display = 'none'; this.current = null; }

  async show(nodeId: string): Promise<void> {
    this.current = nodeId;
    this.el.style.display = 'block';
    this.el.innerHTML = `<div class="mi-loading">Loading ${esc(nodeId)}…</div>`;
    let d: NodeDetail;
    try {
      const res = await apiFetch(`/api/mind/node/${encodeURIComponent(nodeId)}`);
      if (!res.ok) { this.el.innerHTML = `<div class="mi-loading">Node not found.</div>`; return; }
      d = await res.json();
    } catch {
      this.el.innerHTML = `<div class="mi-loading">Could not load node.</div>`;
      return;
    }
    if (this.current !== nodeId) return;
    this.render(d);
  }

  refresh(): void { if (this.current) void this.show(this.current); }

  private render(d: NodeDetail): void {
    const pct = d.usage.percentile === null ? '—' : `${Math.round(d.usage.percentile * 100)}th pct`;
    const conf = Math.round(d.confidence * 100);
    const recs = d.records.map(r => `
      <li class="mi-rec">
        <div class="mi-rec-text">${esc(r.text)}</div>
        <div class="mi-rec-meta">
          <span>conf ${Math.round(r.confidence * 100)}%</span><span>${esc(r.trust)}</span>
          ${r.origin_node ? `<span>from ${esc(r.origin_node)}</span>` : ''}
          ${r.sources.map(s => `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title || new URL(s.url).hostname)}</a>`).join('')}
        </div>
      </li>`).join('');
    const emb = d.records[0]
      ? `<div class="mi-emb"><code>[${d.records[0].embedding_preview.map(x => x.toFixed(3)).join(', ')}, …]</code>
         <span>${d.records[0].embedding_dims} dims · ‖v‖ ${d.records[0].embedding_norm}</span></div>`
      : '<div class="dim">No vectors.</div>';
    const edges = d.edges.slice(0, 30).map(e => `
      <li><span class="mi-rel">${esc(e.relation)}</span> ${e.direction === 'out' ? '→' : '←'}
        <a href="#" data-node="${esc(e.node_id)}">${esc(e.label)}</a></li>`).join('');
    const cur = d.curiosity.map(c => `<li>${esc(c.question)} <span class="dim">(${c.priority.toFixed(2)})</span></li>`).join('');

    this.el.innerHTML = `
      <div class="mi-head">
        <div>
          <div class="mi-kind">${esc(d.kind)} · <span class="mi-tier mi-tier-${esc(d.tier)}">${esc(d.tier)}</span>${d.pinned ? ' · pinned' : ''}</div>
          <div class="mi-title">${esc(d.label)}</div>
        </div>
        <button class="mi-close" title="Close">✕</button>
      </div>
      <div class="mi-grid">
        <div><b>${d.usage.score.toFixed(3)}</b><span>usage score</span></div>
        <div><b>${pct}</b><span>percentile</span></div>
        <div><b>${d.usage.access_count}</b><span>recalls</span></div>
        <div><b>${ago(d.usage.last_accessed)}</b><span>last used</span></div>
      </div>
      <div class="mi-conf"><span>confidence ${conf}%</span><div class="mi-bar"><i style="width:${conf}%"></i></div></div>
      ${d.collection ? `<div class="mi-coll">collection <code>${esc(d.collection.name)}</code> · ${d.collection.record_count} records
        <button class="mi-btn" data-act="records">View vectors</button></div>` : ''}
      ${d.aliases.length ? `<div class="mi-aliases">aka ${d.aliases.map(esc).join(', ')}</div>` : ''}
      <h4>Embedding preview</h4>${emb}
      ${recs ? `<h4>Records</h4><ul class="mi-list">${recs}</ul>` : ''}
      ${edges ? `<h4>Connections</h4><ul class="mi-list mi-edges">${edges}</ul>` : ''}
      ${cur ? `<h4>Curious about</h4><ul class="mi-list">${cur}</ul>` : ''}
      ${['entity', 'abstraction'].includes(d.kind) ? `
      <div class="mi-actions">
        <button class="mi-btn" data-act="pin">${d.pinned ? 'Unpin' : 'Pin'}</button>
        <button class="mi-btn mi-danger" data-act="forget">Forget</button>
      </div>` : ''}
    `;
    this.el.querySelector('.mi-close')?.addEventListener('click', () => this.hide());
    this.el.querySelectorAll<HTMLAnchorElement>('a[data-node]').forEach(a => a.addEventListener('click', ev => {
      ev.preventDefault();
      const id = a.dataset['node']!;
      this.actions.focus(id);
      void this.show(id);
    }));
    this.el.querySelector('[data-act="records"]')?.addEventListener('click', () => {
      if (d.collection) this.actions.showRecords(d.id, d.collection.name);
    });
    this.el.querySelector('[data-act="pin"]')?.addEventListener('click', async () => {
      await apiFetch(`/api/mind/node/${encodeURIComponent(d.id)}/pin`, { method: 'POST', body: JSON.stringify({ pinned: !d.pinned }) });
      this.refresh();
    });
    this.el.querySelector('[data-act="forget"]')?.addEventListener('click', async () => {
      if (!confirm(`Forget "${d.label}" and its ${d.collection?.record_count ?? 0} records? A snapshot from the last sleep can restore it.`)) return;
      const res = await apiFetch(`/api/mind/node/${encodeURIComponent(d.id)}`, { method: 'DELETE' });
      if (res.ok) { this.actions.removed(d.id); this.hide(); }
    });
  }
}
