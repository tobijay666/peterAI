/**
 * RecordsLayer — fly into one node's collection: its individual vectors as a
 * point cloud around the node (positions from the server's PCA of just that
 * collection). Hover a point to read the record; Esc flies back out.
 */
import * as THREE from 'three';
import { apiFetch } from '../api/client';
import type { RecordPoint } from './types';

function dotTexture(): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = c.height = 32;
  const g = c.getContext('2d')!;
  g.beginPath(); g.arc(16, 16, 14, 0, Math.PI * 2); g.fillStyle = '#fff'; g.fill();
  return new THREE.CanvasTexture(c);
}

export class RecordsLayer {
  readonly root = new THREE.Group();
  private points: THREE.Points | null = null;
  private recs: RecordPoint[] = [];
  private tip: HTMLDivElement;
  active: string | null = null;

  constructor(parent: HTMLElement) {
    this.tip = document.createElement('div');
    this.tip.className = 'mind-record-tip';
    this.tip.style.display = 'none';
    parent.appendChild(this.tip);
  }

  async open(nodeId: string, collection: string, center: THREE.Vector3): Promise<number> {
    this.close();
    const res = await apiFetch(`/api/mind/snapshot?view=records&ns=${encodeURIComponent(collection)}`);
    if (!res.ok) return 0;
    const body = await res.json();
    this.recs = ((body.points ?? []) as RecordPoint[]).filter(p => !('demo' in p && (p as any).demo));
    if (!this.recs.length) return 0;
    const n = this.recs.length;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const c = new THREE.Color();
    let maxR = 0;
    for (const r of this.recs) maxR = Math.max(maxR, Math.hypot(...r.position));
    const k = 5.5 / Math.max(1e-6, maxR);
    this.recs.forEach((r, i) => {
      pos.set([center.x + r.position[0] * k, center.y + r.position[1] * k, center.z + r.position[2] * k], i * 3);
      c.setHSL(0.58 - 0.4 * Math.min(1, r.importance), 0.8, 0.6);
      col.set([c.r, c.g, c.b], i * 3);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.5, vertexColors: true, sizeAttenuation: true, toneMapped: false,
      map: dotTexture(), alphaTest: 0.5, transparent: true }));
    this.root.add(this.points);
    this.active = nodeId;
    return n;
  }

  hover(raycaster: THREE.Raycaster, x: number, y: number): void {
    if (!this.points) return;
    raycaster.params.Points = { threshold: 0.25 };
    const hit = raycaster.intersectObject(this.points)[0];
    if (!hit || hit.index === undefined) { this.tip.style.display = 'none'; return; }
    const r = this.recs[hit.index];
    this.tip.textContent = r.document;
    this.tip.style.display = 'block';
    this.tip.style.left = `${x + 14}px`;
    this.tip.style.top = `${y + 10}px`;
  }

  close(): void {
    if (this.points) {
      this.root.remove(this.points);
      this.points.geometry.dispose();
      (this.points.material as THREE.Material).dispose();
      this.points = null;
    }
    this.tip.style.display = 'none';
    this.active = null;
  }
}
