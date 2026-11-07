/**
 * Effects — short-lived visuals driven by mind events.
 *
 *  pulse()     a glowing light that travels a path of points (recall traversal)
 *  dust()      a node dissolving into particles (pruning = active forgetting)
 *  stream()    particles flying from one node to another (embedding migration)
 *  ripple()    an expanding ring at a node (activation / touch)
 */
import * as THREE from 'three';

function glowTexture(): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.25, 'rgba(255,255,255,0.8)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

interface Pulse { sprite: THREE.Sprite; pts: THREE.Vector3[]; seg: number; t: number; segMs: number; }
interface Burst { points: THREE.Points; vel: Float32Array; age: number; life: number; }
interface Stream { points: THREE.Points; from: THREE.Vector3; to: THREE.Vector3; delays: Float32Array; age: number; life: number; }
interface Ripple { mesh: THREE.Mesh; age: number; life: number; }

export class Effects {
  readonly root = new THREE.Group();
  private tex = glowTexture();
  private pulses: Pulse[] = [];
  private bursts: Burst[] = [];
  private streams: Stream[] = [];
  private ripples: Ripple[] = [];
  private ringGeo = new THREE.RingGeometry(0.55, 0.62, 48);

  /** Light travelling along pts; total time = (pts.length - 1) * segMs. */
  pulse(pts: THREE.Vector3[], color = 0xffffff, segMs = 380): number {
    if (pts.length < 2) return 0;
    const mat = new THREE.SpriteMaterial({ map: this.tex, color, transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, toneMapped: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.setScalar(1.6);
    sprite.position.copy(pts[0]);
    this.root.add(sprite);
    this.pulses.push({ sprite, pts: pts.map(p => p.clone()), seg: 0, t: 0, segMs });
    return (pts.length - 1) * segMs;
  }

  dust(at: THREE.Vector3, color: THREE.Color, n = 70): void {
    const pos = new Float32Array(n * 3), vel = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos.set([at.x, at.y, at.z], i * 3);
      const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, r = Math.sqrt(1 - u * u);
      const sp = 0.8 + Math.random() * 1.6;
      vel.set([r * Math.cos(th) * sp, u * sp, r * Math.sin(th) * sp], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ size: 0.22, map: this.tex, color: color.clone().multiplyScalar(0.9),
      transparent: true, opacity: 1, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
    const points = new THREE.Points(geo, mat);
    this.root.add(points);
    this.bursts.push({ points, vel, age: 0, life: 1.4 });
  }

  stream(from: THREE.Vector3, to: THREE.Vector3, count: number, color = 0xb18cff): number {
    const n = Math.max(6, Math.min(60, count * 5));
    const pos = new Float32Array(n * 3), delays = new Float32Array(n);
    for (let i = 0; i < n; i++) { pos.set([from.x, from.y, from.z], i * 3); delays[i] = (i / n) * 0.6; }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ size: 0.28, map: this.tex, color, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, toneMapped: false });
    const points = new THREE.Points(geo, mat);
    this.root.add(points);
    this.streams.push({ points, from: from.clone(), to: to.clone(), delays, age: 0, life: 1.4 });
    return 1400;
  }

  ripple(at: THREE.Vector3, camera: THREE.Camera, color = 0x9ad7ff): void {
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide,
      depthWrite: false, toneMapped: false, blending: THREE.AdditiveBlending });
    const mesh = new THREE.Mesh(this.ringGeo, mat);
    mesh.position.copy(at);
    mesh.quaternion.copy(camera.quaternion);
    this.root.add(mesh);
    this.ripples.push({ mesh, age: 0, life: 0.9 });
  }

  update(dt: number): void {
    // pulses
    this.pulses = this.pulses.filter(p => {
      p.t += (dt * 1000) / p.segMs;
      while (p.t >= 1 && p.seg < p.pts.length - 2) { p.t -= 1; p.seg++; }
      const a = p.pts[p.seg], b = p.pts[p.seg + 1];
      const t = Math.min(1, p.t);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      p.sprite.position.lerpVectors(a, b, e);
      const done = p.seg >= p.pts.length - 2 && p.t >= 1.25;
      (p.sprite.material as THREE.SpriteMaterial).opacity = done ? 0 : 1;
      if (done) { this.root.remove(p.sprite); (p.sprite.material as THREE.Material).dispose(); }
      return !done;
    });
    // bursts
    this.bursts = this.bursts.filter(b => {
      b.age += dt;
      const P = b.points.geometry.getAttribute('position') as THREE.BufferAttribute;
      const arr = P.array as Float32Array;
      for (let i = 0; i < arr.length; i++) { arr[i] += b.vel[i] * dt; b.vel[i] *= 0.97; }
      P.needsUpdate = true;
      (b.points.material as THREE.PointsMaterial).opacity = Math.max(0, 1 - b.age / b.life);
      if (b.age >= b.life) { this.root.remove(b.points); b.points.geometry.dispose(); (b.points.material as THREE.Material).dispose(); return false; }
      return true;
    });
    // streams (gentle arc from -> to)
    this.streams = this.streams.filter(s => {
      s.age += dt;
      const P = s.points.geometry.getAttribute('position') as THREE.BufferAttribute;
      const mid = s.from.clone().lerp(s.to, 0.5).add(new THREE.Vector3(0, s.from.distanceTo(s.to) * 0.25, 0));
      for (let i = 0; i < s.delays.length; i++) {
        const t = Math.max(0, Math.min(1, (s.age - s.delays[i]) / (s.life - 0.6)));
        const a = s.from.clone().lerp(mid, t), b = mid.clone().lerp(s.to, t);
        const q = a.lerp(b, t);
        P.setXYZ(i, q.x, q.y, q.z);
      }
      P.needsUpdate = true;
      if (s.age >= s.life) { this.root.remove(s.points); s.points.geometry.dispose(); (s.points.material as THREE.Material).dispose(); return false; }
      return true;
    });
    // ripples
    this.ripples = this.ripples.filter(r => {
      r.age += dt;
      const t = r.age / r.life;
      r.mesh.scale.setScalar(1 + t * 2.5);
      (r.mesh.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.9 * (1 - t));
      if (t >= 1) { this.root.remove(r.mesh); (r.mesh.material as THREE.Material).dispose(); return false; }
      return true;
    });
  }
}
