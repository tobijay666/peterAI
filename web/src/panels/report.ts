/**
 * Morning report panel — shows what Peter learned while the user was away.
 * Loaded once on app mount; each entry can be dismissed.
 */
import { apiFetch } from '../api/client';

export async function mount(root: HTMLElement): Promise<void> {
  root.innerHTML = `<div class="report-wrap"><h3 class="panel-title">While you were away…</h3><ul id="report-list" class="report-list"><li class="report-loading">Loading…</li></ul></div>`;
  const list = root.querySelector<HTMLUListElement>('#report-list')!;

  try {
    const res  = await apiFetch('/api/report/morning?last_n=20');
    const body = await res.json();
    const entries: any[] = body.entries ?? [];

    if (!entries.length) {
      list.innerHTML = '<li>Nothing new since your last visit.</li>';
      return;
    }

    list.innerHTML = '';
    entries.forEach((entry: any) => {
      const li = document.createElement('li');
      li.className = 'report-item';
      const ts   = entry.ts   ? new Date(entry.ts * 1000).toLocaleString() : '';
      const note = entry.note ?? entry.step ?? JSON.stringify(entry);
      li.innerHTML = `
        <span class="report-ts">${ts}</span>
        <span class="report-note">${escapeHtml(String(note))}</span>
        <button class="report-dismiss">✕</button>
      `;
      li.querySelector('.report-dismiss')?.addEventListener('click', () => li.remove());
      list.appendChild(li);
    });
  } catch {
    list.innerHTML = '<li class="report-error">Could not load report.</li>';
  }

  function escapeHtml(s: string): string {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
}
