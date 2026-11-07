/**
 * Labels — HTML labels for the most-used nodes plus the hovered / focused one,
 * projected from 3-D each frame. Capped so the view stays readable.
 */
import * as THREE from 'three';
import type { GraphLayer } from './graphLayer';

const MAX_LABELS = 18;

export class Labels {
  private el: HTMLDivElement;
  private pool: HTMLDivElement[] = [];
  private v = new THREE.Vector3();
  hovered: string | null = null;
  focused: string | null = null;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'mind-labels';
    parent.appendChild(this.el);
  }

  private label(i: number): HTMLDivElement {
    while (this.pool.length <= i) {
      const d = document.createElement('div');
      d.className = 'mind-label';
      this.el.appendChild(d);
      this.pool.push(d);
    }
    return this.pool[i];
  }

  update(layer: GraphLayer, camera: THREE.Camera, w: number, h: number): void {
    const nodes = layer.allNodes().filter(n => n.kind === 'entity' || n.kind === 'abstraction');
    nodes.sort((a, b) => (b.usage_score ?? 0) - (a.usage_score ?? 0));
    const ids = nodes.slice(0, MAX_LABELS).map(n => n.id);
    for (const extra of [this.hovered, this.focused]) if (extra && !ids.includes(extra)) ids.push(extra);
    let i = 0;
    for (const id of ids) {
      const pos = layer.getPos(id);
      const n = layer.getNode(id);
      if (!pos || !n) continue;
      this.v.copy(pos).project(camera);
      if (this.v.z > 1 || this.v.z < -1) continue;
      const d = this.label(i++);
      d.textContent = n.label;
      d.className = `mind-label tier-${n.tier}${id === this.hovered || id === this.focused ? ' active' : ''}`;
      d.style.transform = `translate(${((this.v.x + 1) / 2) * w}px, ${((1 - this.v.y) / 2) * h - 16}px) translate(-50%, -100%)`;
      d.style.display = 'block';
    }
    for (; i < this.pool.length; i++) this.pool[i].style.display = 'none';
  }
}
