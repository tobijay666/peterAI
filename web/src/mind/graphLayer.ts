/**
 * GraphLayer — the knowledge web: nodes as instanced spheres, edges as line segments.
 *
 * Nodes are grouped by style tier (one InstancedMesh each) because opacity is a
 * material property: high (bright, blooms), moderate, low (dull, faint), fading,
 * faint (curiosity) and fixed (agents, skills, sources). Every node animates
 * toward its target position / scale / colour, so style changes from the server
 * (usage up, fading, pruned) are smooth rather than jumps.
 */
import * as THREE from 'three';
import type { GraphEdge, GraphNode, GraphSnapshot, NodeStyle } from './types';
import { DEFAULT_EDGE_HEX, KIND_HUE, RELATION_HEX } from './types';

type GroupName = 'high' | 'moderate' | 'low' | 'fading' | 'faint' | 'fixed';
const GROUPS: Record<GroupName, { opacity: number }> = {
  high:     { opacity: 1.0 },
  moderate: { opacity: 0.95 },
  low:      { opacity: 0.45 },
  fading:   { opacity: 0.25 },
  faint:    { opacity: 0.45 },
  fixed:    { opacity: 0.95 },
};

interface VNode {
  data: GraphNode;
  pos: THREE.Vector3;
  target: THREE.Vector3;
  scale: number;
  targetScale: number;
  color: THREE.Color;
  targetColor: THREE.Color;
  group: GroupName;
  bornAt: number | null;     // spawn animation start (ms)
  dyingAt: number | null;    // dissolve animation start (ms)
  dieFrom: number;           // scale when the dissolve started
  flash: number;             // extra brightness, decays to 0
}

interface VEdge { data: GraphEdge; progress: number; }

const BASE_RADIUS = 0.42;
const SPAWN_MS = 650;
const DIE_MS = 900;

export function groupFor(style: NodeStyle, kind: string): GroupName {
  if (style.tier === 'high' || style.tier === 'moderate' || style.tier === 'low' || style.tier === 'fading') {
    return style.tier;
  }
  return kind === 'curiosity' ? 'faint' : 'fixed';
}

/** Server style -> render colour. Only the high tier exceeds 1.0, so bloom picks it alone. */
export function colorFor(style: NodeStyle, kind: string, out = new THREE.Color()): THREE.Color {
  const hue = KIND_HUE[kind] ?? 0.6;
  const sat = kind === 'source' ? 0 : Math.max(0, Math.min(1, 0.85 * style.saturation));
  out.setHSL(hue, sat, 0.56);
  const boost = style.tier === 'high' ? 1.0 + style.emissive * 0.22 : Math.min(0.98, 0.72 + style.emissive * 0.5);
  out.multiplyScalar(boost);
  return out;
}

function easeOutBack(t: number): number {
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

export class GraphLayer {
  readonly root = new THREE.Group();
  private nodes = new Map<string, VNode>();
  private edges = new Map<string, VEdge>();
  private meshes = new Map<GroupName, THREE.InstancedMesh>();
  private members = new Map<GroupName, string[]>();
  private capacity = new Map<GroupName, number>();
  private dirty = true;
  private geo = new THREE.SphereGeometry(BASE_RADIUS, 20, 14);
  private lines: THREE.LineSegments;
  private lineCap = 0;
  private dummy = new THREE.Object3D();
  private tmpColor = new THREE.Color();
  private dimmed = 1.0;
  onRemoved: ((id: string, at: THREE.Vector3, color: THREE.Color) => void) | null = null;

  constructor() {
    (Object.keys(GROUPS) as GroupName[]).forEach(g => this.ensureMesh(g, 256));
    const lgeo = new THREE.BufferGeometry();
    const lmat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, toneMapped: false });
    this.lines = new THREE.LineSegments(lgeo, lmat);
    this.lines.frustumCulled = false;
    this.root.add(this.lines);
    this.growLines(512);
  }

  // ---------------------------------------------------------------- buffers

  private ensureMesh(group: GroupName, needed: number): THREE.InstancedMesh {
    const have = this.capacity.get(group) ?? 0;
    let mesh = this.meshes.get(group);
    if (mesh && have >= needed) return mesh;
    const cap = Math.max(needed, have * 2, 256);
    if (mesh) { this.root.remove(mesh); mesh.dispose(); }
    const op = GROUPS[group].opacity;
    const mat = new THREE.MeshBasicMaterial({
      transparent: op < 1, opacity: op, depthWrite: op >= 0.9, toneMapped: false,
    });
    mesh = new THREE.InstancedMesh(this.geo, mat, cap);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.setColorAt(0, new THREE.Color(1, 1, 1));
    mesh.count = 0;
    mesh.frustumCulled = false;
    mesh.userData.group = group;
    this.meshes.set(group, mesh);
    this.capacity.set(group, cap);
    this.root.add(mesh);
    this.dirty = true;
    return mesh;
  }

  private growLines(edgeCap: number): void {
    this.lineCap = edgeCap;
    const geo = this.lines.geometry as THREE.BufferGeometry;
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edgeCap * 6), 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(edgeCap * 6), 3).setUsage(THREE.DynamicDrawUsage));
  }

  // ---------------------------------------------------------------- data in

  setSnapshot(snap: GraphSnapshot): void {
    const keep = new Set(snap.nodes.map(n => n.id));
    for (const id of [...this.nodes.keys()]) if (!keep.has(id)) this.nodes.delete(id);
    for (const n of snap.nodes) this.upsertNode(n, false);
    this.edges.clear();
    for (const e of snap.edges) this.addEdge(e, false);
    this.dirty = true;
  }

  upsertNode(n: Partial<GraphNode> & { id: string }, spawn: boolean): void {
    const existing = this.nodes.get(n.id);
    if (existing) {
      existing.data = { ...existing.data, ...n } as GraphNode;
      if (n.position) existing.target.set(...n.position);
      if (n.style) this.applyStyle(existing, n.style);
      return;
    }
    const data: GraphNode = {
      kind: 'entity', label: n.id, status: 'active', pinned: false, access_count: 1, usage_score: 0,
      percentile: 0.5, tier: 'moderate', collection: null, record_count: 0, confidence: 0.5,
      style: { tier: 'moderate', scale: 1, emissive: 0.4, saturation: 0.85, opacity: 0.95 },
      position: [0, 0, 0], ...n,
    } as GraphNode;
    const pos = new THREE.Vector3(...data.position);
    const v: VNode = {
      data, pos: pos.clone(), target: pos, scale: spawn ? 0 : data.style.scale, targetScale: data.style.scale,
      color: colorFor(data.style, data.kind), targetColor: colorFor(data.style, data.kind),
      group: groupFor(data.style, data.kind), bornAt: spawn ? performance.now() : null, dyingAt: null, dieFrom: 0, flash: spawn ? 1 : 0,
    };
    this.nodes.set(n.id, v);
    this.dirty = true;
  }

  private applyStyle(v: VNode, style: NodeStyle): void {
    v.data.style = style;
    v.data.tier = style.tier;
    v.targetScale = style.scale;
    colorFor(style, v.data.kind, v.targetColor);
    const g = groupFor(style, v.data.kind);
    if (g !== v.group) { v.group = g; this.dirty = true; }
  }

  updateStyle(id: string, style: NodeStyle, extra: Partial<GraphNode> = {}): void {
    const v = this.nodes.get(id);
    if (!v) return;
    v.data = { ...v.data, ...extra };
    this.applyStyle(v, style);
  }

  moveNode(id: string, pos: [number, number, number]): void {
    const v = this.nodes.get(id);
    if (v) v.target.set(...pos);
  }

  flash(id: string, amount = 1): void {
    const v = this.nodes.get(id);
    if (v) v.flash = Math.max(v.flash, amount);
  }

  removeNode(id: string, dissolve = true): void {
    const v = this.nodes.get(id);
    if (!v) return;
    if (!dissolve) { this.finalRemove(id); return; }
    v.dyingAt = performance.now();
    v.dieFrom = v.scale;
    v.targetScale = 0;
  }

  private finalRemove(id: string): void {
    const v = this.nodes.get(id);
    this.nodes.delete(id);
    for (const [eid, e] of this.edges) if (e.data.src === id || e.data.dst === id) this.edges.delete(eid);
    this.dirty = true;
    if (v && this.onRemoved) this.onRemoved(id, v.pos.clone(), v.color.clone());
  }

  addEdge(e: GraphEdge, animate: boolean): void {
    if (this.edges.has(e.id)) return;
    this.edges.set(e.id, { data: e, progress: animate ? 0 : 1 });
  }

  edgeBetween(a: string, b: string): GraphEdge | null {
    for (const e of this.edges.values()) {
      if ((e.data.src === a && e.data.dst === b) || (e.data.src === b && e.data.dst === a)) return e.data;
    }
    return null;
  }

  edge(id: string): GraphEdge | null { return this.edges.get(id)?.data ?? null; }
  getPos(id: string): THREE.Vector3 | null { return this.nodes.get(id)?.pos ?? null; }
  getNode(id: string): GraphNode | null { return this.nodes.get(id)?.data ?? null; }
  getColor(id: string): THREE.Color | null { return this.nodes.get(id)?.color ?? null; }
  allNodes(): GraphNode[] { return [...this.nodes.values()].filter(v => v.dyingAt === null).map(v => v.data); }
  size(): number { return this.nodes.size; }

  /** 0..1 — dim the whole web (sleep mode / records view). */
  setDim(f: number): void { this.dimmed = f; }

  // ---------------------------------------------------------------- picking

  pick(raycaster: THREE.Raycaster): string | null {
    let best: { id: string; d: number } | null = null;
    for (const [g, mesh] of this.meshes) {
      if (mesh.count === 0) continue;
      // three caches an InstancedMesh's bounding sphere on first raycast; nodes move
      // and appear constantly, so recompute it or later picks silently miss.
      mesh.computeBoundingSphere();
      const hits = raycaster.intersectObject(mesh, false);
      const h = hits[0];
      if (h && h.instanceId !== undefined) {
        const id = this.members.get(g)?.[h.instanceId];
        if (id && (!best || h.distance < best.d)) best = { id, d: h.distance };
      }
    }
    return best?.id ?? null;
  }

  // ---------------------------------------------------------------- frame

  update(dt: number): void {
    const now = performance.now();
    const kPos = 1 - Math.exp(-dt * 3.5);
    const kScale = 1 - Math.exp(-dt * 6);
    const kCol = 1 - Math.exp(-dt * 4);
    const dead: string[] = [];

    for (const [id, v] of this.nodes) {
      v.pos.lerp(v.target, kPos);
      if (v.bornAt !== null) {
        const t = Math.min(1, (now - v.bornAt) / SPAWN_MS);
        v.scale = v.targetScale * Math.max(0, easeOutBack(t));
        if (t >= 1) v.bornAt = null;
      } else if (v.dyingAt !== null) {
        const t = Math.min(1, (now - v.dyingAt) / DIE_MS);
        v.scale = v.dieFrom * (1 - t);
        if (t >= 1) dead.push(id);
      } else {
        v.scale += (v.targetScale - v.scale) * kScale;
      }
      v.color.lerp(v.targetColor, kCol);
      v.flash = Math.max(0, v.flash - dt * 1.4);
    }
    for (const id of dead) this.finalRemove(id);

    if (this.dirty) {
      const lists = new Map<GroupName, string[]>();
      (Object.keys(GROUPS) as GroupName[]).forEach(g => lists.set(g, []));
      for (const [id, v] of this.nodes) lists.get(v.group)!.push(id);
      for (const [g, ids] of lists) this.ensureMesh(g, ids.length);
      this.members = lists;
      this.dirty = false;
    }

    for (const [g, ids] of this.members) {
      const mesh = this.meshes.get(g)!;
      mesh.count = ids.length;
      for (let i = 0; i < ids.length; i++) {
        const v = this.nodes.get(ids[i]);
        if (!v) continue;
        const s = Math.max(0.0001, v.scale * (1 + v.flash * 0.35));
        this.dummy.position.copy(v.pos);
        this.dummy.scale.setScalar(s);
        this.dummy.updateMatrix();
        mesh.setMatrixAt(i, this.dummy.matrix);
        this.tmpColor.copy(v.color).multiplyScalar(this.dimmed * (1 + v.flash * 1.6));
        mesh.setColorAt(i, this.tmpColor);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    // edges
    if (this.edges.size > this.lineCap) this.growLines(Math.max(this.edges.size, this.lineCap * 2));
    const geo = this.lines.geometry as THREE.BufferGeometry;
    const P = geo.getAttribute('position') as THREE.BufferAttribute;
    const C = geo.getAttribute('color') as THREE.BufferAttribute;
    let i = 0;
    const c = this.tmpColor;
    for (const e of this.edges.values()) {
      const a = this.nodes.get(e.data.src), b = this.nodes.get(e.data.dst);
      if (!a || !b) continue;
      if (e.progress < 1) e.progress = Math.min(1, e.progress + dt * 2.2);
      const p = e.progress;
      const bx = a.pos.x + (b.pos.x - a.pos.x) * p, by = a.pos.y + (b.pos.y - a.pos.y) * p, bz = a.pos.z + (b.pos.z - a.pos.z) * p;
      P.setXYZ(i * 2, a.pos.x, a.pos.y, a.pos.z);
      P.setXYZ(i * 2 + 1, bx, by, bz);
      c.setHex(RELATION_HEX[e.data.relation] ?? DEFAULT_EDGE_HEX).multiplyScalar(this.dimmed * (p < 1 ? 1.8 : 1));
      C.setXYZ(i * 2, c.r, c.g, c.b);
      C.setXYZ(i * 2 + 1, c.r, c.g, c.b);
      i++;
    }
    geo.setDrawRange(0, i * 2);
    P.needsUpdate = true;
    C.needsUpdate = true;
  }

  dispose(): void {
    for (const m of this.meshes.values()) { m.dispose(); (m.material as THREE.Material).dispose(); }
    this.geo.dispose();
    this.lines.geometry.dispose();
    (this.lines.material as THREE.Material).dispose();
  }
}
