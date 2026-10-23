import type { PeterSocket, MindEvent } from '../live/socket';
import { apiFetch } from '../api/client';

let _lastTaskId: string | null = null;

export function mount(root: HTMLElement, socket: PeterSocket): void {
  root.innerHTML = `
    <div class="chat-panel">
      <div class="chat-header">
        <div class="chat-header-title">
          <div class="chat-avatar">P</div>
          Peter
        </div>
        <div id="chat-status" class="chat-status hidden">Thinking…</div>
      </div>
      <div id="messages" class="messages"></div>
      <form id="chat-form" class="chat-form">
        <div class="chat-input-wrap">
          <textarea id="chat-input" rows="2" placeholder="Ask Peter anything…"></textarea>
        </div>
        <button type="submit" class="btn-send" title="Send (Enter)">↑</button>
      </form>
    </div>
  `;

  const msgEl    = root.querySelector<HTMLElement>('#messages')!;
  const statusEl = root.querySelector<HTMLElement>('#chat-status')!;
  const form     = root.querySelector<HTMLFormElement>('#chat-form')!;
  const input    = root.querySelector<HTMLTextAreaElement>('#chat-input')!;

  // Greeting
  appendMessage('peter', 'Hello! I\'m Peter. I learn and grow with every conversation. What would you like to explore today?');

  // Live thinking indicator
  socket.onEvent((ev: MindEvent) => {
    if (ev.type === 'step_start' || ev.type === 'llm_call' || ev.type === 'tool_call') {
      const detail = ev.type === 'tool_call' ? `using ${(ev as any)['tool']}…`
                   : ev.type === 'llm_call'  ? 'reasoning…'
                   : 'processing…';
      statusEl.textContent = `Peter is ${detail}`;
      statusEl.classList.remove('hidden');
    }
    if (ev.type === 'turn_end') statusEl.classList.add('hidden');
    // Cognitive memory: background research on a knowledge gap
    if (ev.type === 'gap_detected') {
      statusEl.textContent = `Peter is researching "${String(ev['term'] ?? '')}"…`;
      statusEl.classList.remove('hidden');
    }
    if (ev.type === 'research_done') statusEl.classList.add('hidden');
    // The cited follow-up answer once background research has finished
    if (ev.type === 'research_answer' && ev['answer']) {
      appendMessage('peter', String(ev['answer']));
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = '';
    appendMessage('user', text);
    statusEl.textContent = 'Peter is thinking…';
    statusEl.classList.remove('hidden');
    try {
      const res  = await apiFetch('/api/chat', { method: 'POST', body: JSON.stringify({ message: text }) });
      const body = await res.json();
      _lastTaskId = body.task_id ?? null;
      appendMessage('peter', body.response ?? body.detail ?? 'No response.', _lastTaskId);
    } catch {
      appendMessage('peter', '⚠ Network error. Is the server running?');
    } finally {
      statusEl.classList.add('hidden');
    }
  });

  // Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });

  function appendMessage(role: 'user' | 'peter', text: string, taskId?: string | null): void {
    const now  = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    const div  = document.createElement('div');
    div.className = `msg msg-${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    div.appendChild(bubble);

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = role === 'user' ? `You · ${now}` : `Peter · ${now}`;
    div.appendChild(meta);

    if (role === 'peter' && taskId) {
      const fbRow = document.createElement('div');
      fbRow.className = 'feedback-row';
      const goodBtn = document.createElement('button');
      goodBtn.textContent = '👍'; goodBtn.className = 'btn-feedback'; goodBtn.title = 'Good answer';
      const badBtn  = document.createElement('button');
      badBtn.textContent  = '👎'; badBtn.className  = 'btn-feedback'; badBtn.title = 'Bad answer';
      const sentEl  = document.createElement('span');
      sentEl.className = 'feedback-sent hidden'; sentEl.textContent = '✓ Recorded';
      const sendFeedback = async (rating: 'good' | 'bad') => {
        try {
          await apiFetch('/api/feedback', { method:'POST', body: JSON.stringify({ task_id: taskId, rating }) });
          goodBtn.disabled = badBtn.disabled = true;
          sentEl.classList.remove('hidden');
        } catch { /* ignore */ }
      };
      goodBtn.addEventListener('click', () => sendFeedback('good'));
      badBtn.addEventListener('click',  () => sendFeedback('bad'));
      fbRow.append(goodBtn, badBtn, sentEl);
      div.appendChild(fbRow);
    }

    msgEl.appendChild(div);
    msgEl.scrollTop = msgEl.scrollHeight;
  }
}
