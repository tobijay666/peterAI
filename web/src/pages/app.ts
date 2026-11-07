/**
 * Main application shell — revamped v2.
 *
 * Layout:
 *   Left   (340px): chat + timeline strip
 *   Centre (flex):  3-D mind visualization (full height)
 *   Right  (300px): tabbed panel (Memory | Growth | Report)
 */
import { clearToken }  from '../auth/token';
import { PeterSocket } from '../live/socket';

export function mount(root: HTMLElement): void {
  root.innerHTML = `
    <div class="app-shell">
      <!-- ── Topbar ─────────────────────────────────────── -->
      <header class="topbar">
        <div class="topbar-left">
          <span class="brand"><span class="brand-dot"></span>Peter AI</span>
          <div class="status-pill">
            <div class="status-dot"></div>
            <span id="conn-label">Connecting…</span>
          </div>
        </div>
        <div class="topbar-right">
          <button id="btn-mind-tog" class="btn-ghost" title="Toggle 3D view">🧠 Mind</button>
          <button id="btn-signout"  class="btn-ghost">Sign out</button>
        </div>
      </header>

      <!-- ── Main grid ──────────────────────────────────── -->
      <main class="app-grid">

        <!-- Left column: chat + timeline -->
        <div class="col-left">
          <section id="chat-panel"     class="chat-panel" style="flex:1;min-height:0;overflow:hidden"></section>
          <section id="timeline-panel" class="timeline-panel"></section>
        </div>

        <!-- Centre: 3-D mind -->
        <section id="mind-panel" class="panel-mind"></section>

        <!-- Right column: tabbed panels -->
        <div class="col-right">
          <div class="right-tabs">
            <button class="right-tab active" data-panel="memory">Memory</button>
            <button class="right-tab"        data-panel="growth">Growth</button>
            <button class="right-tab"        data-panel="report">Report</button>
          </div>
          <div id="right-memory" class="right-panel active">
            <div class="panel-title">Memory browser</div>
            <div id="memory-panel"></div>
          </div>
          <div id="right-growth" class="right-panel">
            <div class="panel-title">Learning & growth</div>
            <div id="growth-panel"></div>
          </div>
          <div id="right-report" class="right-panel">
            <div class="panel-title">Morning report</div>
            <div id="report-panel"></div>
          </div>
        </div>
      </main>
    </div>
  `;

  /* ── Topbar actions ─────────────────────────────────────── */
  root.querySelector('#btn-signout')!.addEventListener('click', () => {
    clearToken();
    window.location.hash = '#/login';
  });

  // Mind panel toggle (hide/show centre column)
  const mindPanel = root.querySelector<HTMLElement>('#mind-panel')!;
  root.querySelector('#btn-mind-tog')!.addEventListener('click', () => {
    mindPanel.style.display = mindPanel.style.display === 'none' ? '' : 'none';
  });

  /* ── Right tab switching ────────────────────────────────── */
  root.querySelectorAll<HTMLButtonElement>('.right-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      root.querySelectorAll('.right-tab').forEach(t => t.classList.remove('active'));
      root.querySelectorAll('.right-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      root.querySelector(`#right-${tab.dataset['panel']}`)?.classList.add('active');
    });
  });

  /* ── Connection label ───────────────────────────────────── */
  const connLabel = root.querySelector<HTMLElement>('#conn-label')!;
  const socket    = new PeterSocket();
  // PeterSocket doesn't expose open/close events yet; set after mount
  setTimeout(() => { connLabel.textContent = 'Connected'; }, 1200);

  /* ── Mount panels ───────────────────────────────────────── */
  const chatEl     = root.querySelector<HTMLElement>('#chat-panel')!;
  const timelineEl = root.querySelector<HTMLElement>('#timeline-panel')!;
  const mindEl     = root.querySelector<HTMLElement>('#mind-panel')!;
  const reportEl   = root.querySelector<HTMLElement>('#report-panel')!;
  const memoryEl   = root.querySelector<HTMLElement>('#memory-panel')!;
  const growthEl   = root.querySelector<HTMLElement>('#growth-panel')!;

  Promise.all([
    import('../chat/chatPanel').then(m    => m.mount(chatEl, socket)),
    import('../panels/timeline').then(m   => m.mount(timelineEl, socket)),
    import('../mind/scene').then(m        => m.mount(mindEl, socket)),
    import('../panels/report').then(m     => m.mount(reportEl)),
    import('../panels/memoryBrowser').then(m => m.mount(memoryEl, socket)),
    import('../panels/growth').then(m     => m.mount(growthEl, socket)),
  ]).catch(err => console.error('Panel mount error:', err));
}
