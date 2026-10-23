/**
 * Memory browser panel — Phase 7.
 *
 * Features:
 *  • Usage meter: shows total / limit with a coloured bar
 *  • Namespace tabs: one tab per agent namespace, plus "All"
 *  • Search with debounce
 *  • Sort: by importance (desc) or by recency
 *  • Kind filter: all / episode / semantic / finding / skill_trigger / legacy
 *  • Expandable full-text preview on each card
 *  • Per-record: pin/unpin, delete (with confirmation)
 *  • Bulk clear: unpinned only, per namespace or all
 *  • Refreshes when a "learned", "store" or "skill_created" socket event arrives
 */

import { apiFetch } from '../api/client';
import type { PeterSocket, MindEvent } from '../live/socket';

const NAMESPACES = ['peter','researcher','librarian','qa','voice','filesystem','devops'];
const KIND_LABELS: Record<string,string> = {
  episode:       'Episode',
  semantic:      'Semantic',
  finding:       'Finding',
  skill_trigger: 'Skill',
  legacy_finding:'Legacy',
};
const KIND_COLOURS: Record<string,string> = {
  episode:       '#7c9eff',
  semantic:      '#a8ff7c',
  finding:       '#ffe07c',
  skill_trigger: '#ffd700',
  legacy_finding:'#888',
};

interface MemItem {
  id:         string;
  document:   string;
  namespace:  string;
  agent:      string;
  importance: number;
  pinned:     boolean;
  kind:       string;
  score:      number;
}

interface MemStats {
  total:      number;
  limit:      number;
  namespaces: Record<string, number>;
  by_kind:    Record<string, number>;
}

export function mount(root: HTMLElement, socket?: PeterSocket): void {
  root.innerHTML = `
    <div class="mem-browser">
      <h3 class="panel-title">Memory browser</h3>

      <!-- Usage meter -->
      <div class="mem-usage" id="mem-usage">
        <div class="mem-usage-bar"><div class="mem-usage-fill" id="mem-usage-fill"></div></div>
        <span class="mem-usage-label" id="mem-usage-label">Loading…</span>
      </div>

      <!-- Namespace tabs -->
      <div class="mem-tabs" id="mem-tabs">
        <button class="mem-tab active" data-ns="__all__">All</button>
        ${NAMESPACES.map(ns => `<button class="mem-tab" data-ns="${ns}">${ns}</button>`).join('')}
      </div>

      <!-- Controls row -->
      <div class="mem-controls">
        <input  id="mem-query"     class="mem-search-input" type="search" placeholder="Search memories…" />
        <select id="mem-sort"      class="mem-select" title="Sort">
          <option value="score">Relevance</option>
          <option value="importance">Importance</option>
          <option value="recent">Recency</option>
        </select>
        <select id="mem-kind-filt" class="mem-select" title="Kind">
          <option value="">All kinds</option>
          ${Object.entries(KIND_LABELS).map(([k,v]) => `<option value="${k}">${v}</option>`).join('')}
        </select>
      </div>

      <!-- Actions row -->
      <div class="mem-actions">
        <button id="mem-btn-search"  class="btn-sm">Search</button>
        <button id="mem-btn-clear"   class="btn-sm btn-danger" title="Delete all unpinned in current namespace">Clear unpinned</button>
      </div>

      <!-- Results -->
      <ul id="mem-list" class="mem-list"></ul>
    </div>
  `;

  /* ---------- state ---------- */
  let currentNs  = '__all__';
  let lastResults: MemItem[] = [];
  let debounceTimer = 0;

  const queryEl    = root.querySelector<HTMLInputElement>('#mem-query')!;
  const sortEl     = root.querySelector<HTMLSelectElement>('#mem-sort')!;
  const kindFiltEl = root.querySelector<HTMLSelectElement>('#mem-kind-filt')!;
  const listEl     = root.querySelector<HTMLUListElement>('#mem-list')!;
  const usageFill  = root.querySelector<HTMLElement>('#mem-usage-fill')!;
  const usageLabel = root.querySelector<HTMLElement>('#mem-usage-label')!;

  /* ---------- usage meter ---------- */
  async function refreshUsage(): Promise<void> {
    try {
      const res   = await apiFetch('/api/memory/stats');
      if (!res.ok) return;
      const stats = await res.json() as MemStats;
      const pct   = Math.min(100, (stats.total / Math.max(stats.limit, 1)) * 100);
      usageFill.style.width = `${pct}%`;
      usageFill.style.background = pct > 90 ? '#f07050' : pct > 70 ? '#f0c050' : '#50c090';
      usageLabel.textContent = `${stats.total} / ${stats.limit} memories`;
    } catch { /* ignore */ }
  }

  /* ---------- tabs ---------- */
  root.querySelectorAll<HTMLButtonElement>('.mem-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      root.querySelectorAll('.mem-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentNs = btn.dataset['ns'] ?? '__all__';
      doSearch();
    });
  });

  /* ---------- search ---------- */
  async function doSearch(): Promise<void> {
    const q    = queryEl.value.trim();
    const kind = kindFiltEl.value;
    listEl.innerHTML = '<li class="mem-loading">Loading…</li>';

    try {
      // Always fetch up to 100; filtering/sorting happens client-side
      const ns_param = currentNs !== '__all__' ? `&ns=${encodeURIComponent(currentNs)}` : '';
      const url = q
        ? `/api/memory/search?q=${encodeURIComponent(q)}&n=100${ns_param}`
        : `/api/memory/search?q=*&n=100${ns_param}`;
      const res  = await apiFetch(url);
      const body = await res.json();
      let items: MemItem[] = body.results ?? [];

      // Filter by namespace tab (server doesn't support ns filter for /search yet)
      if (currentNs !== '__all__') {
        items = items.filter(i => i.namespace === currentNs);
      }
      // Filter by kind
      if (kind) {
        items = items.filter(i => i.kind === kind);
      }
      // Sort
      const sort = sortEl.value;
      if (sort === 'importance') items.sort((a, b) => b.importance - a.importance);
      else if (sort === 'recent') items.reverse(); // server already returns newest last
      // else: relevance (score) — already sorted by server

      lastResults = items;
      renderList(items);
    } catch {
      listEl.innerHTML = '<li class="mem-error">Search failed.</li>';
    }
  }

  queryEl.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(doSearch, 350);
  });
  root.querySelector('#mem-btn-search')!.addEventListener('click', doSearch);
  root.querySelector('#mem-sort')!.addEventListener('change', () => {
    // Re-sort without re-fetching
    const kind = kindFiltEl.value;
    let items = [...lastResults];
    if (kind) items = items.filter(i => i.kind === kind);
    const sort = sortEl.value;
    if (sort === 'importance') items.sort((a, b) => b.importance - a.importance);
    renderList(items);
  });
  kindFiltEl.addEventListener('change', doSearch);

  /* ---------- bulk clear ---------- */
  root.querySelector('#mem-btn-clear')!.addEventListener('click', async () => {
    const nsLabel = currentNs === '__all__' ? 'ALL namespaces' : `"${currentNs}"`;
    if (!confirm(`Delete ALL unpinned memories in ${nsLabel}?\nThis cannot be undone.`)) return;
    const nsParam = currentNs !== '__all__' ? `?namespace=${encodeURIComponent(currentNs)}` : '';
    try {
      const res  = await apiFetch(`/api/memory/clear${nsParam}`, { method: 'DELETE' });
      const body = await res.json();
      listEl.innerHTML = `<li>Deleted ${body.deleted ?? 0} memories.</li>`;
      refreshUsage();
    } catch {
      listEl.innerHTML = '<li class="mem-error">Clear failed.</li>';
    }
  });

  /* ---------- render ---------- */
  function renderList(items: MemItem[]): void {
    if (!items.length) {
      listEl.innerHTML = '<li class="mem-empty">No memories found.</li>';
      return;
    }
    listEl.innerHTML = '';
    items.forEach(item => {
      const kindLabel  = KIND_LABELS[item.kind] ?? item.kind;
      const kindColour = KIND_COLOURS[item.kind] ?? '#888';
      const preview    = escapeHtml(item.document?.slice(0, 140) ?? '');
      const full       = escapeHtml(item.document ?? '');
      const imp        = (item.importance ?? 0).toFixed(2);
      const pinIcon    = item.pinned ? '📌' : '📍';
      const pinLabel   = item.pinned ? 'Unpin' : 'Pin';

      const li = document.createElement('li');
      li.className = `mem-item ${item.pinned ? 'mem-item--pinned' : ''}`;
      li.innerHTML = `
        <div class="mem-item-head">
          <span class="mem-kind-badge" style="background:${kindColour}22;color:${kindColour};border:1px solid ${kindColour}55">${kindLabel}</span>
          <span class="mem-ns-badge">${item.namespace}</span>
          <span class="mem-imp">imp ${imp}</span>
          ${item.pinned ? '<span class="mem-pin-flag">📌</span>' : ''}
        </div>
        <div class="mem-doc-preview" title="Click to expand">${preview}${item.document?.length > 140 ? '…' : ''}</div>
        <div class="mem-doc-full hidden">${full}</div>
        <div class="mem-item-actions">
          <button class="btn-pin btn-sm" data-id="${item.id}" data-pinned="${item.pinned}">${pinIcon} ${pinLabel}</button>
          <button class="btn-del btn-sm btn-danger" data-id="${item.id}">Delete</button>
        </div>
      `;

      // Expand / collapse on preview click
      const preview_el = li.querySelector<HTMLElement>('.mem-doc-preview')!;
      const full_el    = li.querySelector<HTMLElement>('.mem-doc-full')!;
      preview_el.style.cursor = 'pointer';
      preview_el.addEventListener('click', () => {
        const expanded = !full_el.classList.contains('hidden');
        full_el.classList.toggle('hidden', expanded);
        preview_el.classList.toggle('hidden', !expanded);
      });

      // Pin / unpin
      li.querySelector<HTMLButtonElement>('.btn-pin')!.addEventListener('click', async (e) => {
        const btn    = e.currentTarget as HTMLButtonElement;
        const id     = btn.dataset['id']!;
        const pinned = btn.dataset['pinned'] === 'true';
        await apiFetch('/api/memory/pin', {
          method: 'POST',
          body:   JSON.stringify({ memory_id: id, pinned: !pinned }),
        });
        li.classList.toggle('mem-item--pinned', !pinned);
        btn.dataset['pinned']  = String(!pinned);
        btn.textContent        = (!pinned ? '📌 Unpin' : '📍 Pin');
        li.querySelector<HTMLElement>('.mem-pin-flag')?.remove();
        if (!pinned) {
          const flag = document.createElement('span');
          flag.className   = 'mem-pin-flag';
          flag.textContent = '📌';
          li.querySelector('.mem-item-head')?.appendChild(flag);
        }
        refreshUsage();
      });

      // Delete
      li.querySelector<HTMLButtonElement>('.btn-del')!.addEventListener('click', async (e) => {
        const btn = e.currentTarget as HTMLButtonElement;
        const id  = btn.dataset['id']!;
        if (!confirm('Permanently delete this memory?\nPinned memories cannot be deleted.')) return;
        const res = await apiFetch(`/api/memory/${id}`, { method: 'DELETE' });
        if (res.ok) {
          li.style.opacity    = '0';
          li.style.transition = 'opacity 0.3s';
          setTimeout(() => li.remove(), 300);
          refreshUsage();
        } else {
          const body = await res.json().catch(() => ({}));
          alert(body.detail ?? 'Delete failed (record may be pinned).');
        }
      });

      listEl.appendChild(li);
    });
  }

  function escapeHtml(s: string): string {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  /* ---------- live socket refresh ---------- */
  if (socket) {
    socket.onEvent((ev: MindEvent) => {
      if (ev.type === 'learned' || ev.type === 'store' || ev.type === 'skill_created') {
        refreshUsage();
        // Only re-run search if the query box is empty (don't interrupt active search)
        if (!queryEl.value.trim()) doSearch();
      }
    });
  }

  /* ---------- initial load ---------- */
  refreshUsage();
  doSearch();
}
