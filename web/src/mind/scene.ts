/**
 * Mind view — Peter's knowledge web in 3-D (spec Phase 5).
 *
 *  • Nodes: entity / abstraction / curiosity / skill / agent, sized, coloured and
 *    glowing by the server's usage style (high usage blooms, low usage is dull and
 *    faint, fading nodes shrink, pruned nodes dissolve into dust).
 *  • Recall: the camera flies to the matched node and a light pulse travels from
 *    Peter's core along the exact edges graph recall traversed.
 *  • Sleep: the scene dims; new nodes spawn, edges draw in, embeddings stream into
 *    abstractions, unused nodes fade and are pruned — all from live events.
 *  • Click a node to inspect it; double-click (or "View vectors") to fly into its
 *    collection's point cloud; Esc to come back.
 *
 * three.js is bundled (no CDN global); bloom uses three/examples/jsm.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import type { PeterSocket, MindEvent } from '../live/socket';
import { apiFetch } from '../api/client';
import { CameraRig } from './camera';
import { Effects } from './effects';
import { EventPlayer } from './eventPlayer';
import { GraphLayer } from './graphLayer';
import { Inspector } from './inspector';
import { Labels } from './labels';
import { RecordsLayer } from './records';
import type { GraphEdge, GraphSnapshot, NodeStyle } from './types';

const AWAKE_BG = new THREE.Color(0x060610);
const SLEEP_BG = new THREE.Color(0x040a24);
const CURIOSITY_STYLE: NodeStyle = { tier: 'curiosity', scale: 0.35, emissive: 0.25, saturation: 0.6, opacity: 0.45 };
const SHOWN_KINDS = new Set(['entity', 'abstraction', 'curiosity', 'skill']);
const AGENT_NAMES = new Set(['researcher', 'librarian', 'qa', 'voice', 'filesystem', 'devops']);
const AGENT_HEX: Record<string, number> = {
  peter: 0x7c9eff, researcher: 0xff9f7c, librarian: 0xa8ff7c, filesystem: 0xffe07c,
  devops: 0xe07cff, qa: 0x7cefff, voice: 0xff7cca,
};

type MindState = 'awake' | 'thinking' | 'researching' | 'sleeping';

export async function mount(root: HTMLElement, socket: PeterSocket): Promise<void> {
  root.classList.add('mind-root');
  root.innerHTML = '';

  // ── HUD ──────────────────────────────────────────────────── //
  const hud = document.createElement('div');
  hud.className = 'mind-hud';
  hud.innerHTML = `
    <div class="mind-hud-left">
      <span class="mind-hud-title">Mind</span>
      <span class="mind-state" data-state="awake">awake</span>
      <span class="mind-hud-stats"></span>
    </div>
    <div class="mind-hud-right">
      <button class="mind-btn" data-act="home" title="Reset view">⌂</button>
      <button class="mind-btn" data-act="sleep" title="Run a sleep cycle now">Sleep now</button>
    </div>`;
  root.appendChild(hud);
  const stateEl = hud.querySelector<HTMLElement>('.mind-state')!;
  const statsEl = hud.querySelector<HTMLElement>('.mind-hud-stats')!;

  const toast = document.createElement('div');
  toast.className = 'mind-toast';
  root.appendChild(toast);
  const hint = document.createElement('div');
  hint.className = 'mind-hint';
  hint.textContent = 'No knowledge yet — ask Peter about something and watch it grow.';
  root.appendChild(hint);
  const legend = document.createElement('div');
  legend.className = 'mind-legend';
  legend.innerHTML = `
    <span><i class="lg lg-entity"></i>topic</span><span><i class="lg lg-abstraction"></i>abstraction</span>
    <span><i class="lg lg-curiosity"></i>curiosity</span><span class="dim">bright = often recalled · faint = fading</span>`;
  root.appendChild(legend);

  // ── renderer / scene ─────────────────────────────────────── //
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(1, 1);
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.domElement.className = 'mind-canvas';
  root.insertBefore(renderer.domElement, hud);

  const scene = new THREE.Scene();
  const bg = AWAKE_BG.clone();
  let bgTarget = AWAKE_BG;
  scene.background = bg;
  scene.fog = new THREE.FogExp2(bg.getHex(), 0.012);

  const rig = new CameraRig(renderer.domElement);
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, rig.camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.75, 0.35, 0.95);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // Peter's core: where every recall pulse starts
  const coreMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0x7c9eff).multiplyScalar(1.3), toneMapped: false });
  const core = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 24), coreMat);
  scene.add(core);
  let coreFlash = 0;

  // Lines from Peter's core to each main subject ("Writing", "Investing"); the
  // topics then hang below their subject. Updated every frame so they follow nodes.
  let coreLinks: string[] = [];
  const MAX_CORE_LINKS = 64;
  const coreLinkPos = new Float32Array(MAX_CORE_LINKS * 6);
  const coreLinkGeo = new THREE.BufferGeometry();
  coreLinkGeo.setAttribute('position', new THREE.BufferAttribute(coreLinkPos, 3));
  coreLinkGeo.setDrawRange(0, 0);
  const coreLinkLines = new THREE.LineSegments(coreLinkGeo, new THREE.LineBasicMaterial({
    color: new THREE.Color(0x7c9eff).multiplyScalar(1.1), transparent: true, opacity: 0.75, toneMapped: false }));
  coreLinkLines.frustumCulled = false;
  scene.add(coreLinkLines);
  function updateCoreLinks(): void {
    let k = 0;
    for (const id of coreLinks) {
      const p = layer.getPos(id);
      if (!p || k >= MAX_CORE_LINKS) continue;
      coreLinkPos.set([0, 0, 0, p.x, p.y, p.z], k * 6);
      k++;
    }
    coreLinkGeo.setDrawRange(0, k * 2);
    (coreLinkGeo.attributes['position'] as THREE.BufferAttribute).needsUpdate = true;
  }
  const coreBase = new THREE.Color(0x7c9eff).multiplyScalar(1.3);
  const coreFlashColor = new THREE.Color();

  const layer = new GraphLayer();
  const effects = new Effects();
  const records = new RecordsLayer(root);
  scene.add(layer.root, effects.root, records.root);
  const labels = new Labels(root);
  const coreLabel = document.createElement('div');
  coreLabel.className = 'mind-label mind-core-label';
  coreLabel.textContent = 'Peter';
  root.appendChild(coreLabel);
  const coreProj = new THREE.Vector3();
  layer.onRemoved = (_id, at, color) => effects.dust(at, color);

  const inspector = new Inspector(root, {
    focus: id => { const p = layer.getPos(id); if (p) rig.focus(p); labels.focused = id; },
    showRecords: (id, coll) => void openRecords(id, coll),
    removed: id => layer.removeNode(id, true),
  });

  // ── state / HUD helpers ─────────────────────────────────── //
  let state: MindState = 'awake';
  function setState(s: MindState): void {
    state = s;
    stateEl.textContent = s;
    stateEl.dataset['state'] = s;
    if (s === 'sleeping') { bgTarget = SLEEP_BG; layer.setDim(0.6); }
    else { bgTarget = AWAKE_BG; if (!records.active) layer.setDim(1); }
  }
  let toastTimer = 0;
  function say(text: string, ms = 2600): void {
    toast.textContent = text;
    toast.classList.add('show');
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove('show'), ms);
  }
  function updateStats(): void {
    const all = layer.allNodes();
    const topics = all.filter(n => n.kind === 'entity').length;
    const abs = all.filter(n => n.kind === 'abstraction').length;
    const cur = all.filter(n => n.kind === 'curiosity').length;
    statsEl.textContent = `${topics} topics · ${abs} abstractions · ${cur} questions`;
    hint.style.display = topics + abs === 0 ? 'block' : 'none';
  }
  function flashCore(hex: number, amount = 1): void { coreFlashColor.setHex(hex).multiplyScalar(1.6); coreFlash = amount; }

  // ── data ─────────────────────────────────────────────────── //
  // "tree": connected topics as top-down tree diagrams (server-computed positions).
  // In tree mode the layout depends on the whole graph, so structural events
  // trigger a debounced re-fetch and nodes glide to their new places.
  let layoutMode = 'tree';
  // tree (flat diagrams) and cluster (3-D, similar topics close) are laid out by the server
  function serverLaid(): boolean { return layoutMode === 'tree' || layoutMode === 'cluster'; }
  let firstFit = true;
  let relayoutTimer = 0;
  function scheduleRelayout(ms = 900): void {
    if (!serverLaid()) return;
    window.clearTimeout(relayoutTimer);
    relayoutTimer = window.setTimeout(() => void fetchSnapshot(), ms);
  }

  async function fetchSnapshot(): Promise<void> {
    try {
      const res = await apiFetch('/api/mind/snapshot?view=graph');
      if (!res.ok) { console.error('[Mind] snapshot failed', res.status, await res.text().catch(() => '')); return; }
      const snap = await res.json();
      if (snap.view !== 'graph' || !Array.isArray(snap.nodes)) {
        hint.textContent = 'Cognitive memory is off (config/cognition.yaml: enabled: false).';
        hint.style.display = 'block';
        return;
      }
      layer.setSnapshot(snap as GraphSnapshot);
      layoutMode = snap.layout ?? 'semantic';
      coreLinks = serverLaid() && Array.isArray(snap.core_links) ? snap.core_links : [];
      // flat tree: re-frame on every change; 3-D cluster: frame once, then leave the camera to the user
      if (serverLaid() && (layoutMode === 'tree' || firstFit)) {
        rig.fitTo((snap as GraphSnapshot).nodes.map(n => n.position), firstFit ? 0 : 900, layoutMode === 'tree');
        firstFit = false;
      }
      updateStats();
    } catch (err) {
      console.warn('[Mind] snapshot error', err);
    }
  }

  async function openRecords(nodeId: string, collection: string): Promise<void> {
    const center = layer.getPos(nodeId);
    if (!center) return;
    const n = await records.open(nodeId, collection, center);
    if (!n) { say('This collection has no vectors yet.'); return; }
    layer.setDim(0.25);
    rig.focus(center, 9, 900);
    say(`${n} vectors in ${collection} — hover to read, Esc to go back`, 4000);
  }
  function closeRecords(): void {
    if (!records.active) return;
    records.close();
    layer.setDim(state === 'sleeping' ? 0.6 : 1);
  }

  function nearPos(anchorId: string | null | undefined, r = 2.2): [number, number, number] {
    const a = anchorId ? layer.getPos(anchorId) : null;
    const base = a ?? new THREE.Vector3(0, 0, 0);
    const t = Math.random() * Math.PI * 2, u = Math.random() * 2 - 1, s = Math.sqrt(1 - u * u);
    const rr = a ? r : 6 + Math.random() * 4;
    return [base.x + rr * s * Math.cos(t), base.y + rr * u, base.z + rr * s * Math.sin(t)];
  }

  // ── event handling (sequenced by the player) ─────────────── //
  function onTraverse(ev: MindEvent): number {
    const path = (ev['path'] as string[] | undefined) ?? [];
    const edgeIds = (ev['edges'] as string[] | undefined) ?? [];
    if (!path.length) return 0;
    const order = new Map(path.map((id, i) => [id, i] as [string, number]));
    const viaEdge = new Set<string>();
    const hops: Array<[string, string]> = [];
    for (const eid of edgeIds) {
      const e = layer.edge(eid);
      if (!e) continue;
      const [a, b] = (order.get(e.src) ?? 99) <= (order.get(e.dst) ?? 99) ? [e.src, e.dst] : [e.dst, e.src];
      viaEdge.add(b);
      hops.push([a, b]);
    }
    const matched = path.filter(id => !viaEdge.has(id) && layer.getPos(id));
    const origin = new THREE.Vector3(0, 0, 0);
    let first = 0;
    for (const id of matched) {
      first = Math.max(first, effects.pulse([origin, layer.getPos(id)!], 0xdff3ff, 520));
      layer.flash(id, 1);
    }
    window.setTimeout(() => {
      for (const [a, b] of hops) {
        const pa = layer.getPos(a), pb = layer.getPos(b);
        if (pa && pb) { effects.pulse([pa, pb], 0xc9b3ff, 420); layer.flash(b, 0.7); }
      }
    }, first);
    if (matched[0] && !records.active) {
      const p = layer.getPos(matched[0]);
      if (p) { rig.focus(p, 15, 800); labels.focused = matched[0]; }
    }
    flashCore(0xdff3ff, 0.8);
    return first + (hops.length ? 460 : 0) + 200;
  }

  function handle(ev: MindEvent): number {
    switch (ev.type) {
      case 'turn_start': case 'llm_call': case 'tool_call':
        if (state === 'awake') setState('thinking');
        return 0;
      case 'turn_end':
        if (state === 'thinking') setState('awake');
        return 0;
      case 'traverse':
        return onTraverse(ev);
      case 'node_touched': {
        const id = ev['node_id'] as string;
        if (ev['style']) layer.updateStyle(id, ev['style'] as NodeStyle, {
          access_count: Number(ev['access_count'] ?? 0), usage_score: Number(ev['usage_score'] ?? 0),
          percentile: Number(ev['percentile'] ?? 0.5),
        });
        const p = layer.getPos(id);
        if (p) effects.ripple(p, rig.camera);
        if (inspector.nodeId === id) inspector.refresh();
        return 120;
      }
      case 'gap_detected': {
        setState('researching');
        const term = String(ev['term'] ?? '');
        say(`Knowledge gap: "${term}" — researching…`);
        const protoId = `gap_${term.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
        if (!layer.getNode(protoId)) {
          layer.upsertNode({
            id: protoId,
            kind: 'entity',
            label: term,
            style: { tier: 'low', scale: 0.65, emissive: 0.8, saturation: 0.7, opacity: 0.75 },
            position: nearPos(null, 3.5),
            collection: null,
          }, true);
          updateStats();
        }
        return 200;
      }
      case 'research_step': {
        try { say(`Reading ${new URL(String(ev['url'])).hostname}…`, 1800); } catch { /* ignore */ }
        flashCore(0xff9f7c, 0.7);
        const nid = (ev['node_id'] as string | undefined) ?? (ev['topic_node_id'] as string | undefined);
        const p = nid ? layer.getPos(nid) : null;
        if (nid && p) {
          effects.pulse([new THREE.Vector3(0, 0, 0), p], 0xff9f7c, 400);
          layer.flash(nid, 0.6);
        }
        return 160;
      }
      case 'node_spawned': {
        // Only knowledge is drawn; agents (researcher, librarian…) and tasks are plumbing
        if (!SHOWN_KINDS.has(String(ev['kind'] ?? 'entity'))) return 0;
        // Helper agents (researcher, librarian…) are plumbing: never drawn, even as a topic node
        if (AGENT_NAMES.has(String(ev['label'] ?? '').trim().toLowerCase())) return 0;
        const id = ev['node_id'] as string;
        const label = String(ev['label'] ?? '');
        const protoId = `gap_${label.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
        if (layer.getNode(protoId) && protoId !== id) {
          layer.removeNode(protoId, false);
        }
        // Tree mode: the node grows out of Peter's core, then the re-layout moves it into its tree
        const pos = serverLaid() ? [0, -2, 0] as [number, number, number]
          : (ev['position'] as [number, number, number] | null) ?? nearPos(null);
        layer.upsertNode({ id, kind: String(ev['kind'] ?? 'entity'), label,
          style: ev['style'] as NodeStyle, position: pos, collection: (ev['collection'] as string) ?? null }, true);
        updateStats();
        scheduleRelayout();
        if (state !== 'sleeping' && !serverLaid()) { const p = layer.getPos(id); if (p) rig.focus(p, 18, 700); }
        return 550;
      }
      case 'research_done': {
        const id = ev['node_id'] as string | null;
        if (id) {
          if (ev['position'] && !serverLaid()) layer.moveNode(id, ev['position'] as [number, number, number]);
          scheduleRelayout(300);
          if (ev['style']) layer.updateStyle(id, ev['style'] as NodeStyle);
          layer.flash(id, 1);
          say(`Learned ${ev['records']} facts about "${ev['topic']}"`);
        }
        if (state === 'researching') setState('awake');
        return 300;
      }
      case 'edge_drawn': {
        const e: GraphEdge = { id: ev['edge_id'] as string, src: ev['src'] as string, dst: ev['dst'] as string,
          relation: ev['relation'] as string };
        if (!layer.getPos(e.src) || !layer.getPos(e.dst)) return 0;   // e.g. source nodes are not drawn
        layer.addEdge(e, true);
        scheduleRelayout();
        return 260;
      }
      case 'curiosity_added': {
        const id = (ev['node_id'] as string) ?? `cur_${ev['id']}`;
        const topic = ev['topic_node_id'] as string | null;
        const pos = nearPos(topic, 2.2);
        layer.upsertNode({ id, kind: 'curiosity', label: String(ev['question'] ?? ''), style: CURIOSITY_STYLE,
          position: pos }, true);
        if (topic) {
          layer.addEdge({ id: `${id}->${topic}`, src: id, dst: topic, relation: 'curious_about' }, true);
          const topPos = layer.getPos(topic);
          const curPos = layer.getPos(id);
          if (topPos && curPos) effects.pulse([topPos, curPos], 0xffe07c, 320);
        }
        updateStats();
        scheduleRelayout();
        return 240;
      }
      case 'sleep_start':
        closeRecords();
        setState('sleeping');
        rig.goHome();
        say('Sleeping — consolidating memory', 3500);
        return 700;
      case 'sleep_phase':
        say(`Sleep: ${String(ev['name']).replace(/_/g, ' ')}`, 2000);
        return 350;
      case 'embedding_migrated': {
        const a = layer.getPos(ev['from_node'] as string), b = layer.getPos(ev['to_node'] as string);
        return a && b ? effects.stream(a, b, Number(ev['count'] ?? 4)) * 0.8 : 0;
      }
      case 'node_fading':
        if (ev['style']) layer.updateStyle(ev['node_id'] as string, ev['style'] as NodeStyle, { status: 'fading' });
        return 450;
      case 'node_pruned': {
        const id = ev['node_id'] as string;
        layer.flash(id, 0.6);
        layer.removeNode(id, true);
        if (inspector.nodeId === id) inspector.hide();
        window.setTimeout(updateStats, 1000);
        scheduleRelayout(1100);
        return 950;
      }
      case 'layout_updated': {
        if (serverLaid()) { scheduleRelayout(); return 0; }
        for (const m of (ev['moves'] as Array<{ node_id: string; position: [number, number, number] }>) ?? []) {
          layer.moveNode(m.node_id, m.position);
        }
        return 1200;
      }
      case 'sleep_end':
        setState('awake');
        say('Awake — memory reorganised');
        window.setTimeout(() => void fetchSnapshot(), 1500);
        return 300;
      case 'store': case 'learned':
        flashCore(0x00ff88, 0.7);
        return 0;
      case 'skill_created':
        flashCore(0xffd966, 1);
        return 0;
      case 'delegate':
        flashCore(AGENT_HEX[String(ev['agent'] ?? 'peter')] ?? 0x888888, 0.8);
        return 0;
      default:
        return 0;
    }
  }

  const player = new EventPlayer(handle);
  socket.onEvent(ev => player.push(ev));

  // ── interaction ──────────────────────────────────────────── //
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  function setMouse(e: MouseEvent): void {
    const r = renderer.domElement.getBoundingClientRect();
    mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(mouse, rig.camera);
  }
  renderer.domElement.addEventListener('mousemove', e => {
    setMouse(e);
    if (records.active) {
      const r = renderer.domElement.getBoundingClientRect();
      records.hover(raycaster, e.clientX - r.left, e.clientY - r.top);
      return;
    }
    const id = layer.pick(raycaster);
    labels.hovered = id;
    renderer.domElement.style.cursor = id ? 'pointer' : '';
  });
  // A single click waits briefly so a double-click can claim the gesture
  // (otherwise the click's camera fly-to moves the node out from under the cursor).
  let clickTimer = 0;
  renderer.domElement.addEventListener('click', e => {
    setMouse(e);
    const id = layer.pick(raycaster);
    window.clearTimeout(clickTimer);
    if (!id) return;
    clickTimer = window.setTimeout(() => {
      labels.focused = id;
      const p = layer.getPos(id);
      if (p) rig.focus(p, 14, 700);
      void inspector.show(id);
    }, 260);
  });
  renderer.domElement.addEventListener('dblclick', e => {
    window.clearTimeout(clickTimer);
    setMouse(e);
    const id = layer.pick(raycaster);
    const n = id ? layer.getNode(id) : null;
    if (n?.collection) void openRecords(n.id, n.collection);
  });
  window.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (records.active) { closeRecords(); rig.goHome(); } else inspector.hide();
  });
  hud.querySelector('[data-act="home"]')!.addEventListener('click', () => { closeRecords(); rig.goHome(); });
  hud.querySelector('[data-act="sleep"]')!.addEventListener('click', async () => {
    const res = await apiFetch('/api/mind/sleep', { method: 'POST', body: JSON.stringify({ mode: 'active' }) });
    const body = await res.json().catch(() => ({}));
    say(body.started ? 'Going to sleep…' : `Not now: ${body.reason ?? body.detail ?? 'unavailable'}`);
  });

  // ── frame loop ───────────────────────────────────────────── //
  let w = 1, h = 1;
  function onResize(): void {
    w = root.clientWidth; h = root.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    composer.setSize(w, h);
    bloom.setSize(w, h);
    rig.resize(w, h);
  }
  new ResizeObserver(onResize).observe(root);
  requestAnimationFrame(onResize);

  const clock = new THREE.Clock();
  let t = 0;
  function frame(): void {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, clock.getDelta());
    t += dt;
    bg.lerp(bgTarget, 1 - Math.exp(-dt * 2));
    (scene.fog as THREE.FogExp2).color.copy(bg);
    core.scale.setScalar(1 + 0.05 * Math.sin(t * 1.7) + coreFlash * 0.3);
    coreMat.color.copy(coreBase).lerp(coreFlashColor, coreFlash);
    coreFlash = Math.max(0, coreFlash - dt * 1.5);
    if (state === 'thinking' || state === 'researching') core.rotation.y += dt * 2;
    layer.update(dt);
    updateCoreLinks();
    effects.update(dt);
    rig.update(dt);
    labels.update(layer, rig.camera, w, h);
    coreProj.set(0, 0, 0).project(rig.camera);
    coreLabel.style.transform = `translate(${((coreProj.x + 1) / 2) * w}px, ${((1 - coreProj.y) / 2) * h + 14}px) translate(-50%, 0)`;
    composer.render();
  }
  // Debug / automation hook (used by the headless UI check); read-only in practice.
  (window as unknown as { __peterMind?: unknown }).__peterMind = {
    layer, rig, player,
    screenPos(id: string): [number, number] | null {
      const p = layer.getPos(id);
      if (!p) return null;
      const v = p.clone().project(rig.camera);
      const r = renderer.domElement.getBoundingClientRect();
      return [r.left + ((v.x + 1) / 2) * r.width, r.top + ((1 - v.y) / 2) * r.height];
    },
  };

  frame();
  await fetchSnapshot();
}
