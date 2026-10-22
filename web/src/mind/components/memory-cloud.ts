/// <reference path="../../env.d.ts" />
/**
 * A-Frame component: memory-cloud
 *
 * Renders each memory as a point cloud using three.js Points (more
 * performant than InstancedMesh for thousands of small dots).
 *
 * Animations:
 *   recall   → target points flash white, then a thin beam appears from
 *              the origin (Peter core) to each recalled point for 800 ms
 *   store    → new point pops in with a brief scale pulse; snapshot
 *              re-fetched after 500 ms so the position is correct
 *   delegate → origin pulses with the agent's colour
 *
 * Inspector: clicking a point emits a custom 'node-click' event on the
 * parent entity with the memory's metadata so scene.ts can show the
 * inspector panel.
 *
 * NOTE: THREE is a CDN global (window.THREE), not a module import.
 */


const AGENT_COLOURS: Record<string, string> = {
  peter:      '#7c9eff',
  researcher: '#ff9f7c',
  librarian:  '#a8ff7c',
  filesystem: '#ffe07c',
  devops:     '#e07cff',
  qa:         '#7cefff',
  voice:      '#ff7cca',
};
const DEFAULT_COLOUR = '#888888';

const agentColor = (agent: string): string =>
  AGENT_COLOURS[agent] ?? DEFAULT_COLOUR;

interface MemoryPoint {
  id:         string;
  position:   [number, number, number];
  importance: number;
  agent:      string;
  document:   string;
  kind:       string;
  trust:      string;
  pinned:     boolean;
  demo?:      boolean;  // true for seed/placeholder points shown before real memories exist
}

// Store the full point data for the inspector
const _pointData: MemoryPoint[] = [];

AFRAME.registerComponent('memory-cloud', {
  schema: {},

  init(this: any) {
    this._points      = [] as MemoryPoint[];
    this._idToIndex   = new Map<string, number>();
    this._pointsObj   = null as any;   // THREE.Points
    this._beams       = [] as any[];   // THREE.Line[]
    this._raycaster   = new THREE.Raycaster();
    this._mouse       = new THREE.Vector2();
    this._camera      = null as any;

    // Beam geometry reuse pool
    this._beamMat = new THREE.LineBasicMaterial({
      color: 0x7c9eff,
      transparent: true,
      opacity: 0.7,
    });

    this.el.addEventListener('snapshot', (e: CustomEvent) => {
      this._applySnapshot(e.detail?.points ?? []);
    });
    this.el.addEventListener('mind-event', (e: CustomEvent) => {
      this._handleEvent(e.detail);
    });

    // Click → ray-cast for inspector
    this.el.sceneEl?.addEventListener('click', (e: MouseEvent) => this._onClick(e));
  },

  // -----------------------------------------------------------------
  _applySnapshot(points: MemoryPoint[]) {
    _pointData.length = 0;
    this._points = points;
    this._idToIndex = new Map(points.map((p: MemoryPoint, i: number) => [p.id, i]));
    _pointData.push(...points);
    this._rebuild();
  },

  _rebuild() {
    // Remove old object
    if (this._pointsObj) {
      this.el.object3D.remove(this._pointsObj);
      this._pointsObj.geometry.dispose();
      this._pointsObj.material.dispose();
    }
    this._clearBeams();

    const n = this._points.length;
    if (n === 0) return;

    const positions = new Float32Array(n * 3);
    const colors    = new Float32Array(n * 3);
    const sizes     = new Float32Array(n);
    const col       = new THREE.Color();

    this._points.forEach((p: MemoryPoint, i: number) => {
      positions[i * 3]     = p.position[0];
      positions[i * 3 + 1] = p.position[1];
      positions[i * 3 + 2] = p.position[2];

      col.set(agentColor(p.agent));
      colors[i * 3]     = col.r;
      colors[i * 3 + 1] = col.g;
      colors[i * 3 + 2] = col.b;

      sizes[i] = Math.max(4, p.importance * 12);
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size',     new THREE.BufferAttribute(sizes, 1));

    // Demo points are all at the start of the array only when every point is demo
    const allDemo = this._points.every((p: MemoryPoint) => p.demo);
    const mat = new THREE.PointsMaterial({
      size:            allDemo ? 0.25 : 0.35,
      vertexColors:    true,
      sizeAttenuation: true,
      transparent:     true,
      opacity:         allDemo ? 0.35 : 0.92,
    });

    this._pointsObj = new THREE.Points(geo, mat);
    this._pointsObj.renderOrder = 1;
    this.el.object3D.add(this._pointsObj);
  },

  // -----------------------------------------------------------------
  _handleEvent(ev: any) {
    switch (ev.type) {
      case 'recall':
        this._animateRecall(ev.memory_ids ?? []);
        break;
      case 'store':
        // Re-fetch snapshot after a short delay so the new point gets placed
        setTimeout(() => this._refetchSnapshot(), 500);
        break;
      case 'delegate':
        this._pulsePeterCore(ev.agent ?? 'peter');
        break;
      case 'learned':
        // Phase 6: green growth glow + re-fetch snapshot to add new point
        this._pulseLearn(ev.memory_id);
        setTimeout(() => this._refetchSnapshot(), 600);
        break;
      case 'skill_created':
        // Gentle gold pulse on Peter core to signal a new skill was born
        this._pulsePeterCore('skill');
        break;
    }
  },

  // Growth pulse: flash a bright green glow from the Peter core sphere
  _pulseLearn(_memId: string) {
    // Pulse the Peter-core entity if present
    const core = this.el.sceneEl?.querySelector('#peter-core') as any;
    if (!core || !core.object3D) return;
    const mesh = core.object3D.children[0];
    if (!mesh?.material) return;
    const orig = mesh.material.emissive?.clone() ?? new THREE.Color(0);
    mesh.material.emissive = new THREE.Color(0x00ff88);
    mesh.material.emissiveIntensity = 0.9;
    setTimeout(() => {
      mesh.material.emissive = orig;
      mesh.material.emissiveIntensity = 0.2;
    }, 700);
  },

  // Recall animation: flash points white + draw beams from origin
  _animateRecall(ids: string[]) {
    if (!this._pointsObj) return;
    const geo    = this._pointsObj.geometry;
    const colors = geo.getAttribute('color') as any;

    const origin = new THREE.Vector3(0, 0, 0);
    const savedColors: [number, number, number, number][] = [];

    const col = new THREE.Color();
    ids.forEach((id: string) => {
      const idx = this._idToIndex.get(id);
      if (idx === undefined) return;

      // Save original colour
      savedColors.push([idx, colors.getX(idx), colors.getY(idx), colors.getZ(idx)]);

      // Flash white
      colors.setXYZ(idx, 1, 1, 1);

      // Draw a beam from origin to this point
      const p = this._points[idx];
      if (p) {
        const target = new THREE.Vector3(...p.position);
        const beamGeo = new THREE.BufferGeometry().setFromPoints([origin, target]);
        const beam = new THREE.Line(beamGeo, this._beamMat);
        this.el.object3D.add(beam);
        this._beams.push(beam);
      }
    });

    colors.needsUpdate = true;

    // Restore after 800 ms
    setTimeout(() => {
      savedColors.forEach(([idx, r, g, b]) => {
        colors.setXYZ(idx, r, g, b);
      });
      colors.needsUpdate = true;
      this._clearBeams();
    }, 800);
  },

  // Pulse the Peter core sphere with the agent's colour
  _pulsePeterCore(agent: string) {
    const core = document.getElementById('peter-core');
    if (!core) return;
    const hex = agentColor(agent);
    core.setAttribute('color', hex);
    setTimeout(() => core.setAttribute('color', '#7c9eff'), 600);
  },

  _clearBeams() {
    this._beams.forEach((b: any) => {
      this.el.object3D.remove(b);
      b.geometry.dispose();
    });
    this._beams = [];
  },

  // Re-fetch snapshot from the API when new memories are stored
  async _refetchSnapshot() {
    try {
      const { apiFetch } = await import('../../api/client');
      const res  = await apiFetch('/api/mind/snapshot');
      if (res.ok) {
        const snap = await res.json();
        this._applySnapshot(snap.points ?? []);
      }
    } catch { /* silently skip */ }
  },

  // Click ray-casting for inspector
  _onClick(e: MouseEvent) {
    if (!this._pointsObj) return;

    const canvas = this.el.sceneEl?.canvas as HTMLCanvasElement;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    this._mouse.x =  ((e.clientX - rect.left)  / rect.width)  * 2 - 1;
    this._mouse.y = -((e.clientY - rect.top)   / rect.height) * 2 + 1;

    // Get the A-Frame camera
    if (!this._camera) {
      const camEl = document.getElementById('mind-camera');
      this._camera = (camEl as any)?.object3D?.children?.[0] ?? null;
    }
    if (!this._camera) return;

    this._raycaster.setFromCamera(this._mouse, this._camera);
    const hits = this._raycaster.intersectObject(this._pointsObj);
    if (!hits.length) return;

    const idx   = hits[0].index ?? -1;
    const point = _pointData[idx];
    if (!point) return;

    this.el.dispatchEvent(new CustomEvent('node-click', { bubbles: true, detail: point }));
  },
});
