/**
 * Camera rig — OrbitControls plus smooth focus tweens (recall: "animate camera focus").
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface Tween { fromPos: THREE.Vector3; toPos: THREE.Vector3; fromTarget: THREE.Vector3; toTarget: THREE.Vector3; t: number; ms: number; }

export class CameraRig {
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private tween: Tween | null = null;
  private home = { pos: new THREE.Vector3(0, 8, 34), target: new THREE.Vector3(0, 0, 0) };

  constructor(dom: HTMLElement) {
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 400);
    this.camera.position.copy(this.home.pos);
    this.controls = new OrbitControls(this.camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 200;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.25;
    this.controls.addEventListener('start', () => { this.tween = null; this.controls.autoRotate = false; });
  }

  /** Fly so that `target` is centred, `distance` away, over `ms`. */
  focus(target: THREE.Vector3, distance = 14, ms = 800): void {
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    if (dir.lengthSq() < 1e-6) dir.set(0, 0.3, 1).normalize();
    this.tween = {
      fromPos: this.camera.position.clone(), toPos: target.clone().add(dir.multiplyScalar(distance)),
      fromTarget: this.controls.target.clone(), toTarget: target.clone(), t: 0, ms,
    };
  }

  /** Frame a set of points: new home = their centre, at a distance that fits them.
   *  flat: a 2-D tree diagram, viewed face-on without spinning; otherwise a 3-D
   *  cloud, viewed from slightly above while it slowly turns. */
  fitTo(points: Array<[number, number, number]>, ms = 900, flat = true): void {
    const all = points.concat([[0, 0, 0]]);    // keep Peter's core in view
    const vFov = (this.camera.fov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    if (flat) {
      this.controls.autoRotate = false;
      const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const centre = new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, 0);
      const dist = Math.max(18, (maxY - minY + 8) / 2 / Math.tan(vFov / 2), (maxX - minX + 8) / 2 / Math.tan(hFov / 2));
      this.home = { pos: centre.clone().add(new THREE.Vector3(0, 0, Math.min(dist, 150))), target: centre };
    } else {
      this.controls.autoRotate = true;
      // Peter stays the centre; fit the sphere that holds every node
      const r = Math.max(8, ...all.map(p => Math.hypot(p[0], p[1], p[2]))) + 3;
      const dist = r / Math.sin(Math.min(vFov, hFov) / 2);
      const dir = new THREE.Vector3(0, 0.35, 1).normalize();
      this.home = { pos: dir.multiplyScalar(Math.min(dist, 150)), target: new THREE.Vector3(0, 0, 0) };
    }
    if (ms <= 0) {
      this.tween = null;
      this.camera.position.copy(this.home.pos);
      this.controls.target.copy(this.home.target);
    } else {
      this.goHome(ms);
    }
  }

  goHome(ms = 900): void {
    this.tween = { fromPos: this.camera.position.clone(), toPos: this.home.pos.clone(),
      fromTarget: this.controls.target.clone(), toTarget: this.home.target.clone(), t: 0, ms };
  }

  resize(w: number, h: number): void {
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  update(dt: number): void {
    if (this.tween) {
      const tw = this.tween;
      tw.t = Math.min(1, tw.t + (dt * 1000) / tw.ms);
      const e = 1 - Math.pow(1 - tw.t, 3);
      this.camera.position.lerpVectors(tw.fromPos, tw.toPos, e);
      this.controls.target.lerpVectors(tw.fromTarget, tw.toTarget, e);
      if (tw.t >= 1) this.tween = null;
    }
    this.controls.update();
  }
}
