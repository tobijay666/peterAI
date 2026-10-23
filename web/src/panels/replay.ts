/**
 * Replay panel — fetches the event log and replays it into the mind view
 * by dispatching events through the same channel the live socket uses.
 */
import { apiFetch }             from '../api/client';
import type { PeterSocket, MindEvent } from '../live/socket';

export function mount(root: HTMLElement, socket: PeterSocket): void {
  root.innerHTML = `
    <div class="replay-wrap">
      <button id="btn-replay" class="btn-replay">▶ Replay last session</button>
      <p id="replay-status" class="dim" style="padding:0.3rem 0;font-size:0.75rem"></p>
    </div>
  `;

  const btn    = root.querySelector<HTMLButtonElement>('#btn-replay')!;
  const status = root.querySelector<HTMLParagraphElement>('#replay-status')!;

  btn.addEventListener('click', async () => {
    btn.disabled    = true;
    status.textContent = 'Loading…';
    try {
      const res    = await apiFetch('/api/mind/replay?last_n=300');
      const body   = await res.json();
      const events = (body.events ?? []) as MindEvent[];
      if (!events.length) { status.textContent = 'No events to replay.'; return; }
      status.textContent = `Replaying ${events.length} events…`;

      // Drive listeners exactly as the live socket does
      const listeners = (socket as any).listeners as Array<(e: MindEvent) => void>;
      for (const ev of events) {
        listeners.forEach(fn => { try { fn(ev); } catch { /* ignore */ } });
        await new Promise(r => setTimeout(r, 80));
      }
      status.textContent = '✓ Replay complete';
    } catch {
      status.textContent = '✗ Replay failed';
    } finally {
      btn.disabled = false;
    }
  });
}
