/**
 * Timeline panel — shows live events for the current conversation turn
 * as a D3-drawn scrolling list.  It is displayed below the chat panel.
 */
import type { PeterSocket, MindEvent } from '../live/socket';
import * as d3 from 'd3';

interface TimelineRow {
  ts:   number;
  type: string;
  label: string;
}

const TYPE_LABELS: Record<string, string> = {
  turn_start:       '▶ Turn started',
  turn_end:         '■ Turn ended',
  llm_call:         '🤖 LLM call',
  tool_call:        '🔧 Tool call',
  delegate:         '👥 Delegated to',
  recall:           '🔍 Recalled memories',
  store:            '💾 Stored memory',
  context_injected: '📌 Context injected',
  step_start:       '⚙ Step started',
  step_end:         '✓ Step done',
};

export function mount(root: HTMLElement, socket: PeterSocket): void {
  root.innerHTML = `<div class="timeline-wrap"><h3 class="panel-title">Live events</h3><ul id="tl-list" class="tl-list"></ul></div>`;
  const list = root.querySelector<HTMLUListElement>('#tl-list')!;

  socket.onEvent((ev: MindEvent) => {
    const label = TYPE_LABELS[ev.type] ?? ev.type;
    const extra = ev.type === 'delegate'    ? ` ${ev['agent']}` :
                  ev.type === 'tool_call'   ? ` ${ev['tool']}`  :
                  ev.type === 'recall'      ? ` (${(ev['memory_ids'] as string[] | undefined)?.length ?? 0} mem)` :
                  '';
    const li = document.createElement('li');
    li.className = `tl-row tl-${ev.type}`;
    const time = new Date((ev['ts'] as number) * 1000).toLocaleTimeString();
    li.textContent = `${time}  ${label}${extra}`;
    list.appendChild(li);
    list.scrollTop = list.scrollHeight;
    // Keep only the last 60 rows to avoid unbounded growth
    while (list.children.length > 60) list.removeChild(list.firstChild!);
  });
}
