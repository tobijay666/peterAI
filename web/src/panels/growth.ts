/**
 * Memory growth panel — shows live memory stats as clean stat cards.
 *
 * Phase 6: polls /api/memory/stats every 30 s so the numbers update as
 * Peter learns during the session.
 */
import { apiFetch }     from '../api/client';
import type { PeterSocket, MindEvent } from '../live/socket';

// How often (ms) to re-fetch stats while the panel is open
const POLL_INTERVAL = 30_000;

interface MemoryStats {
  total:      number;
  limit:      number;
  namespaces: Record<string, number>;
  by_kind:    Record<string, number>;
}

export function mount(root: HTMLElement, socket?: PeterSocket): void {
  root.innerHTML = `
    <div class="growth-wrap">
      <h3 class="panel-title">Memory growth</h3>

      <div class="growth-usage-row">
        <span class="growth-usage-label">Total memories</span>
        <span class="growth-usage-value"><span id="growth-total">–</span> <span class="growth-usage-limit" id="growth-limit">/ –</span></span>
      </div>
      <div class="growth-progress-wrap">
        <div class="growth-progress-bar" id="growth-bar" style="width:0%"></div>
      </div>

      <div class="growth-section-title">By namespace</div>
      <div id="growth-ns" class="growth-ns-list"></div>

      <div class="growth-section-title">By kind</div>
      <div id="growth-kind" class="growth-kind-list"></div>
    </div>
  `;

  const totalEl  = root.querySelector<HTMLElement>('#growth-total')!;
  const limitEl  = root.querySelector<HTMLElement>('#growth-limit')!;
  const barEl    = root.querySelector<HTMLElement>('#growth-bar')!;
  const nsEl     = root.querySelector<HTMLElement>('#growth-ns')!;
  const kindEl   = root.querySelector<HTMLElement>('#growth-kind')!;

  async function refresh(): Promise<void> {
    try {
      const res = await apiFetch('/api/memory/stats');
      if (!res.ok) return;
      const stats = await res.json() as MemoryStats;
      renderStats(stats);
    } catch { /* network hiccup — ignore */ }
  }

  function renderStats(stats: MemoryStats): void {
    const { total, limit, namespaces, by_kind } = stats;

    // Usage row
    totalEl.textContent = String(total);
    limitEl.textContent = `/ ${limit}`;
    const pct = limit > 0 ? Math.min(100, Math.round((total / limit) * 100)) : 0;
    barEl.style.width = `${pct}%`;
    const nearLimit = total > limit * 0.9;
    barEl.style.background = nearLimit
      ? 'linear-gradient(90deg, #f0a050, #e05030)'
      : 'linear-gradient(90deg, var(--accent), var(--accent-2))';

    // Namespace rows
    nsEl.innerHTML = '';
    const sortedNs = Object.entries(namespaces).sort((a, b) => b[1] - a[1]);
    const nsMax = sortedNs[0]?.[1] ?? 1;
    sortedNs.forEach(([ns, count]) => {
      if (count === 0) return;
      const pctNs = Math.max(4, Math.round((count / Math.max(nsMax, 1)) * 100));
      const row = document.createElement('div');
      row.className = 'growth-ns-row';
      row.innerHTML = `
        <span class="growth-ns-name">${ns}</span>
        <div class="growth-ns-bar-wrap">
          <div class="growth-ns-bar" style="width:${pctNs}%"></div>
        </div>
        <span class="growth-ns-count">${count}</span>
      `;
      nsEl.appendChild(row);
    });

    // Kind badges
    kindEl.innerHTML = '';
    const sortedKinds = Object.entries(by_kind).sort((a, b) => b[1] - a[1]);
    sortedKinds.slice(0, 8).forEach(([kind, count]) => {
      const span = document.createElement('span');
      span.className   = `kind-badge kind-${kind}`;
      span.textContent = `${kind} ×${count}`;
      kindEl.appendChild(span);
    });
  }

  // Initial load + periodic poll
  refresh();
  const tid = window.setInterval(refresh, POLL_INTERVAL);

  // Live event: refresh immediately on each "learned" event
  if (socket) {
    socket.onEvent((ev: MindEvent) => {
      if (ev.type === 'learned' || ev.type === 'store' || ev.type === 'skill_created') {
        refresh();
      }
    });
  }

  // Cleanup on panel unmount (best-effort)
  root.addEventListener('remove', () => clearInterval(tid), { once: true });
}
