/// <reference path="../env.d.ts" />
/**
 * Mind view — Pure Three.js scene (no A-Frame).
 *
 * Renders:
 *  • Memory cloud  — one glowing Point per memory, coloured by agent
 *  • Peter core    — pulsing central sphere
 *  • Recall beams  — lines from core to recalled points (800 ms flash)
 *  • Thinking ring — animated torus when Peter is processing
 *  • Live labels   — HUD overlay (canvas 2-D) for memory count / state
 *
 * Works entirely with window.THREE (CDN global, r128).
 * No imports from 'three' — avoids Vite bundling conflicts.
 */
import type { PeterSocket, MindEvent } from '../live/socket';
import { apiFetch } from '../api/client';

/* ── types ──────────────────────────────────────────────────── */
interface MemPoint {
  id:         string;
  document:   string;
  agent:      string;
  importance: number;
  kind:       string;
  trust:      string;
  pinned:     boolean;
  demo:       boolean;
  position:   [number, number, number];
}

const AGENT_HEX: Record<string, number> = {
  peter:      0x7c9eff,
  researcher: 0xff9f7c,
  librarian:  0xa8ff7c,
  filesystem: 0xffe07c,
  devops:     0xe07cff,
  qa:         0x7cefff,
  voice:      0xff7cca,
};
const DEFAULT_HEX = 0x888888;

/* ── main export ─────────────────────────────────────────────── */
export async function mount(root: HTMLElement, socket: PeterSocket): Promise<void> {
  // Container must fill parent
  root.style.cssText = 'position:relative;width:100%;height:100%;overflow:hidden;background:#060610';

  // Loading overlay
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:12px;
    background:#060610;color:#7c9eff;font-family:Inter,sans-serif;
    font-size:13px;z-index:10;
  `;
  overlay.innerHTML = `
    <div style="width:32px;height:32px;border:3px solid #1a2255;border-top-color:#7c9eff;
      border-radius:50%;animation:spin 0.9s linear infinite"></div>
    <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
    <span>Initialising mind view…</span>
  `;
  root.appendChild(overlay);

  // HUD label bar
  const hud = document.createElement('div');
  hud.style.cssText = `
    position:absolute;top:12px;left:0;right:0;display:flex;
    justify-content:space-between;align-items:center;padding:0 16px;
    pointer-events:none;z-index:5;font-family:Inter,sans-serif;
  `;
  hud.innerHTML = `
    <span style="font-size:13px;font-weight:600;color:#c8d5ff;letter-spacing:.04em">🧠 Mind</span>
    <span id="mind-stats" style="font-size:11px;color:#7c9eff;opacity:.8"></span>
  `;
  root.appendChild(hud);
  const statsEl = hud.querySelector<HTMLElement>('#mind-stats')!;

  // Node inspector panel
  const inspector = document.createElement('div');
  inspector.style.cssText = `
    position:absolute;bottom:16px;left:16px;right:16px;max-height:120px;
    background:rgba(13,13,31,.92);border:1px solid #1e2a4a;border-radius:10px;
    padding:10px 14px;font-family:Inter,sans-serif;font-size:11px;color:#c8d5ff;
    display:none;backdrop-filter:blur(8px);z-index:5;
  `;
  root.appendChild(inspector);

  // ── Three.js setup ────────────────────────────────────────── //
  const T = (window as any).THREE as typeof THREE;

  const W = root.clientWidth  || 600;
  const H = root.clientHeight || 500;

  const renderer = new T.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x060610, 1);
  renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%!important;height:100%!important;';
  root.appendChild(renderer.domElement);

  const scene  = new T.Scene();
  scene.fog    = new T.FogExp2(0x060610, 0.018);

  const camera = new T.PerspectiveCamera(60, W / H, 0.1, 200);
  camera.position.set(0, 4, 32);

  // Lights
  scene.add(new T.AmbientLight(0x1a2255, 1.4));
  const pt1 = new T.PointLight(0x7c9eff, 1.2, 60, 1.5);
  pt1.position.set(0, 10, 5);
  scene.add(pt1);
  const pt2 = new T.PointLight(0xa78bfa, 0.5, 40, 2);
  pt2.position.set(-15, 5, -10);
  scene.add(pt2);

  // Peter core sphere
  const coreMat = new T.MeshPhongMaterial({
    color: 0x7c9eff, emissive: 0x3050cc, emissiveIntensity: 0.4,
    shininess: 90, transparent: true, opacity: 0.95,
  });
  const coreMesh = new T.Mesh(new T.SphereGeometry(0.7, 32, 32), coreMat);
  scene.add(coreMesh);

  // Glow ring around core
  const ringGeo = new T.TorusGeometry(1.2, 0.04, 12, 64);
  const ringMat = new T.MeshBasicMaterial({ color: 0x7c9eff, transparent: true, opacity: 0.3 });
  const ring    = new T.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  scene.add(ring);

  // Thinking ring (shown while Peter processes)
  const thinkGeo = new T.TorusGeometry(2.2, 0.06, 8, 48);
  const thinkMat = new T.MeshBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: 0 });
  const thinkRing = new T.Mesh(thinkGeo, thinkMat);
  scene.add(thinkRing);

  // ── Orbit controls (manual, no dep needed) ───────────────── //
  let isDragging = false, prevMX = 0, prevMY = 0;
  let theta = 0, phi = Math.PI / 3, radius = 32;

  function updateCamera() {
    camera.position.set(
      radius * Math.sin(phi) * Math.sin(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.cos(theta),
    );
    camera.lookAt(0, 0, 0);
  }
  updateCamera();

  const canvas = renderer.domElement;
  canvas.addEventListener('mousedown', (e: MouseEvent) => { isDragging = true; prevMX = e.clientX; prevMY = e.clientY; });
  window.addEventListener('mouseup',   () => { isDragging = false; });
  window.addEventListener('mousemove', (e: MouseEvent) => {
    if (!isDragging) return;
    theta -= (e.clientX - prevMX) * 0.006;
    phi    = Math.max(0.15, Math.min(Math.PI - 0.15, phi - (e.clientY - prevMY) * 0.006));
    prevMX = e.clientX; prevMY = e.clientY;
    updateCamera();
  });
  canvas.addEventListener('wheel', (e: WheelEvent) => {
    radius = Math.max(6, Math.min(80, radius + e.deltaY * 0.04));
    updateCamera();
    e.preventDefault();
  }, { passive: false });

  // ── Point cloud state ─────────────────────────────────────── //
  let points: MemPoint[] = [];
  let pointsObj: any = null;
  let beams: any[] = [];

  const beamMat = new T.LineBasicMaterial({ color: 0x7c9eff, transparent: true, opacity: 0.7 });

  function buildPointCloud(pts: MemPoint[]) {
    // remove old
    if (pointsObj) { scene.remove(pointsObj); pointsObj.geometry.dispose(); }
    clearBeams();
    points = pts;
    if (!pts.length) return;

    const n         = pts.length;
    const positions = new Float32Array(n * 3);
    const colors    = new Float32Array(n * 3);
    const col       = new T.Color();
    const allDemo   = pts.every(p => p.demo);

    pts.forEach((p, i) => {
      positions[i * 3]     = p.position[0];
      positions[i * 3 + 1] = p.position[1];
      positions[i * 3 + 2] = p.position[2];
      col.setHex(AGENT_HEX[p.agent] ?? DEFAULT_HEX);
      colors[i * 3]     = col.r;
      colors[i * 3 + 1] = col.g;
      colors[i * 3 + 2] = col.b;
    });

    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new T.BufferAttribute(colors, 3));

    const mat = new T.PointsMaterial({
      size: allDemo ? 0.28 : 0.42,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: allDemo ? 0.38 : 0.92,
    });

    pointsObj = new T.Points(geo, mat);
    scene.add(pointsObj);

    const real = pts.filter(p => !p.demo).length;
    statsEl.textContent = allDemo
      ? 'Showing preview — start chatting to grow real memories'
      : `${real} memor${real === 1 ? 'y' : 'ies'}`;
    statsEl.style.opacity = allDemo ? '0.45' : '0.8';
  }

  function clearBeams() {
    beams.forEach(b => { scene.remove(b); b.geometry.dispose(); });
    beams = [];
  }

  function flashRecall(ids: string[]) {
    if (!pointsObj) return;
    const colAttr = pointsObj.geometry.getAttribute('color') as any;
    const saved: [number, number, number, number][] = [];
    const origin = new T.Vector3(0, 0, 0);

    ids.forEach(id => {
      const idx = points.findIndex(p => p.id === id);
      if (idx < 0) return;
      saved.push([idx, colAttr.getX(idx), colAttr.getY(idx), colAttr.getZ(idx)]);
      colAttr.setXYZ(idx, 1, 1, 1);
      const p   = points[idx];
      const end = new T.Vector3(...p.position);
      const bg  = new T.BufferGeometry().setFromPoints([origin, end]);
      const ln  = new T.Line(bg, beamMat);
      scene.add(ln);
      beams.push(ln);
    });
    colAttr.needsUpdate = true;

    setTimeout(() => {
      saved.forEach(([idx, r, g, b]) => colAttr.setXYZ(idx, r, g, b));
      colAttr.needsUpdate = true;
      clearBeams();
    }, 850);
  }

  function pulseCore(hex: number, duration = 600) {
    coreMat.emissive.setHex(hex);
    coreMat.emissiveIntensity = 1.0;
    setTimeout(() => { coreMat.emissive.setHex(0x3050cc); coreMat.emissiveIntensity = 0.4; }, duration);
  }

  async function fetchSnapshot() {
    try {
      const res = await apiFetch('/api/mind/snapshot');
      if (!res.ok) return;
      const snap = await res.json();
      buildPointCloud(snap.points ?? []);
    } catch { /* backend not ready */ }
  }

  // ── Click → inspector ─────────────────────────────────────── //
  const raycaster = new T.Raycaster();
  raycaster.params.Points = { threshold: 0.5 };
  const mouse = new T.Vector2();

  canvas.addEventListener('click', (e: MouseEvent) => {
    if (!pointsObj) return;
    const rect = canvas.getBoundingClientRect();
    mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(pointsObj);
    if (!hits.length) { inspector.style.display = 'none'; return; }
    const p = points[hits[0].index ?? -1];
    if (!p) return;
    inspector.style.display = 'block';
    inspector.innerHTML = `
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-weight:600;color:#7c9eff">${p.agent}</span>
        <span style="color:#4a5568;cursor:pointer" id="insp-close">✕</span>
      </div>
      <div style="margin-bottom:6px;line-height:1.5;color:#c8d5ff">${p.document}</div>
      <div style="color:#4a6080;font-size:10px">
        ${p.kind} · importance ${p.importance.toFixed(2)} · ${p.trust}${p.pinned ? ' · 📌' : ''}
        ${p.demo ? ' · <em>demo point</em>' : ''}
      </div>
    `;
    inspector.querySelector('#insp-close')?.addEventListener('click', () => { inspector.style.display = 'none'; });
  });

  // ── Animation loop ─────────────────────────────────────────── //
  let t = 0;
  let thinking = false;

  function animate() {
    requestAnimationFrame(animate);
    t += 0.016;

    // Core breathe
    const s = 1 + 0.04 * Math.sin(t * 1.6);
    coreMesh.scale.setScalar(s);
    coreMesh.rotation.y += 0.004;

    // Ring spin
    ring.rotation.z += 0.003;

    // Thinking ring
    if (thinking) {
      thinkRing.rotation.z += 0.035;
      thinkRing.rotation.x += 0.012;
      thinkMat.opacity = 0.55 + 0.25 * Math.sin(t * 4);
    } else {
      thinkMat.opacity = Math.max(0, thinkMat.opacity - 0.04);
    }

    // Point cloud slow drift
    if (pointsObj) pointsObj.rotation.y += 0.0008;

    renderer.render(scene, camera);
  }

  // ── Resize ────────────────────────────────────────────────── //
  const ro = new ResizeObserver(() => {
    const w = root.clientWidth, h = root.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  ro.observe(root);

  // ── Socket events ─────────────────────────────────────────── //
  socket.onEvent((ev: MindEvent) => {
    switch (ev.type) {
      case 'thinking':
      case 'step_start':
        thinking = true;
        pulseCore(0xa78bfa, 200);
        break;
      case 'turn_end':
        thinking = false;
        break;
      case 'recall':
        flashRecall((ev as any).memory_ids ?? []);
        break;
      case 'store':
      case 'learned':
        pulseCore(0x00ff88, 500);
        setTimeout(fetchSnapshot, 700);
        break;
      case 'skill_created':
        pulseCore(0xffd966, 700);
        break;
      case 'delegate':
        pulseCore(AGENT_HEX[(ev as any).agent ?? 'peter'] ?? DEFAULT_HEX, 400);
        break;
    }
  });

  // ── Boot ──────────────────────────────────────────────────── //
  overlay.style.display = 'none';
  animate();
  await fetchSnapshot();
}
