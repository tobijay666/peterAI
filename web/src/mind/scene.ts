/**
 * Mind view — 3-D A-Frame scene.
 * Revamped v2: proper overlay, label bar, stats counter, fixed canvas sizing.
 */
import type { PeterSocket, MindEvent } from '../live/socket';
import { apiFetch } from '../api/client';

import './components/memory-cloud';
import './components/knowledge-graph';

export async function mount(root: HTMLElement, socket: PeterSocket): Promise<void> {
  // A-Frame needs to fill the parent completely
  root.style.position = 'relative';
  root.style.overflow = 'hidden';
  root.style.width    = '100%';
  root.style.height   = '100%';

  root.innerHTML = `
    <!-- Loading overlay -->
    <div id="mind-overlay" class="mind-overlay">
      <div class="mind-overlay-spinner"></div>
      <span>Initialising mind view…</span>
    </div>

    <!-- Label bar — on top of the scene, pointer-events:none -->
    <div class="mind-label-bar">
      <span class="mind-title">🧠 Mind</span>
      <span class="mind-stats" id="mind-stats">0 memories</span>
    </div>

    <!-- A-Frame scene -->
    <a-scene
      id="mind-scene"
      embedded
      renderer="antialias: true; colorManagement: true; physicallyCorrectLights: true"
      vr-mode-ui="enabled: false"
      background="color: #060610"
      style="width:100%;height:100%;display:block"
    >
      <!-- Orbit camera -->
      <a-entity
        id="mind-camera"
        camera
        look-controls="pointerLockEnabled: false; reverseMouseDrag: false"
        wasd-controls="enabled: false"
        position="0 2 40"
      ></a-entity>

      <!-- Memory cloud (point cloud) -->
      <a-entity id="memory-cloud" memory-cloud></a-entity>

      <!-- Knowledge graph (force-directed edges) -->
      <a-entity id="knowledge-graph" knowledge-graph></a-entity>

      <!-- Central Peter core -->
      <a-sphere
        id="peter-core"
        radius="0.7"
        color="#7c9eff"
        opacity="0.95"
        position="0 0 0"
        animation="property: rotation; to: 0 360 0; loop: true; dur: 18000; easing: linear"
        animation__pulse="property: scale; from: 1 1 1; to: 1.08 1.08 1.08; loop: true; dur: 2000; direction: alternate; easing: easeInOutSine"
      ></a-sphere>

      <!-- Glow ring around core -->
      <a-torus
        radius="1.1" radius-tubular="0.03"
        color="#7c9eff" opacity="0.25"
        rotation="90 0 0"
        animation="property: rotation; from: 90 0 0; to: 90 360 0; loop: true; dur: 12000; easing: linear"
      ></a-torus>

      <!-- Lighting -->
      <a-light type="ambient"  color="#1a2255" intensity="1.2"></a-light>
      <a-light type="point"    color="#7c9eff" intensity="1.0" position="0 10 5" decay="1" distance="40"></a-light>
      <a-light type="point"    color="#a78bfa" intensity="0.4" position="-15 5 -10" decay="2" distance="30"></a-light>

      <a-sky color="#060610"></a-sky>
    </a-scene>

    <!-- Node inspector -->
    <div id="mind-inspector" class="mind-inspector hidden"></div>
  `;

  const overlay   = root.querySelector<HTMLElement>('#mind-overlay')!;
  const scene     = root.querySelector<Element>('#mind-scene')!;
  const inspector = root.querySelector<HTMLElement>('#mind-inspector')!;
  const statsEl   = root.querySelector<HTMLElement>('#mind-stats')!;

  // Wait for A-Frame to boot
  await new Promise<void>(resolve => {
    if ((scene as any).hasLoaded) { resolve(); return; }
    scene.addEventListener('loaded', () => resolve(), { once: true });
  });

  overlay.classList.add('hidden');

  const cloudEl = document.getElementById('memory-cloud');
  const graphEl = document.getElementById('knowledge-graph');

  // Fetch initial snapshot
  try {
    const res = await apiFetch('/api/mind/snapshot');
    if (res.ok) {
      const snap = await res.json();
      cloudEl?.dispatchEvent(new CustomEvent('snapshot', { detail: snap }));
      graphEl?.dispatchEvent(new CustomEvent('snapshot', { detail: snap }));
      const allPoints  = snap?.points ?? [];
      const realPoints = allPoints.filter((p: any) => !p.demo);
      const count      = realPoints.length;
      const isDemo     = count === 0 && allPoints.length > 0;
      statsEl.textContent = isDemo
        ? 'No memories yet — start chatting!'
        : `${count} memor${count === 1 ? 'y' : 'ies'}`;
      if (isDemo) statsEl.style.opacity = '0.5';
    }
  } catch {
    // Server not yet ready — components start empty
  }

  // Forward events
  let memCount = 0;
  socket.onEvent((ev: MindEvent) => {
    cloudEl?.dispatchEvent(new CustomEvent('mind-event', { detail: ev }));
    graphEl?.dispatchEvent(new CustomEvent('mind-event', { detail: ev }));
    if (ev.type === 'learned' || ev.type === 'store') {
      memCount++;
      statsEl.textContent = `${memCount} memor${memCount === 1 ? 'y' : 'ies'}`;
    }
    if (ev.type === 'recall') showRecallToast(root, ev);
  });

  // Inspector
  cloudEl?.addEventListener('node-click', (e: Event) => {
    const detail = (e as CustomEvent).detail;
    if (!detail) { inspector.classList.add('hidden'); return; }
    inspector.innerHTML = `
      <div class="inspector-content">
        <button class="inspector-close" id="insp-close">✕</button>
        <div class="inspector-tag">${detail.kind ?? 'memory'} · ${detail.agent ?? ''}</div>
        <div class="inspector-text">${escHtml(detail.document ?? '')}</div>
        <div class="inspector-meta">
          Importance: ${(detail.importance ?? 0).toFixed(2)} ·
          Trust: ${detail.trust ?? 'unknown'}
          ${detail.pinned ? ' · 📌 Pinned' : ''}
        </div>
      </div>
    `;
    inspector.classList.remove('hidden');
    inspector.querySelector('#insp-close')?.addEventListener('click', () => {
      inspector.classList.add('hidden');
    });
  });
}

function showRecallToast(root: HTMLElement, ev: MindEvent): void {
  const ids = (ev as any)['memory_ids'] as string[] | undefined;
  if (!ids?.length) return;
  let toast = root.querySelector<HTMLElement>('#recall-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id        = 'recall-toast';
    toast.className = 'recall-toast';
    root.appendChild(toast);
  }
  toast.textContent = `🔍 Recalled ${ids.length} memor${ids.length === 1 ? 'y' : 'ies'}`;
  toast.classList.remove('hidden', 'fade-out');
  setTimeout(() => toast!.classList.add('fade-out'), 1800);
  setTimeout(() => toast!.classList.add('hidden'),   2300);
}

function escHtml(s: string): string {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
