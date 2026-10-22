/**
 * Ambient type declarations for CDN globals and packages
 * that lack complete type definitions.
 */

// Three.js loaded from CDN — available as window.THREE
declare const THREE: typeof import('three');


// d3-force-3d has no @types package; declare enough to satisfy TS
declare module 'd3-force-3d' {
  export function forceSimulation(nodes?: any[]): any;
  export function forceLink(links?: any[]): any;
  export function forceManyBody(): any;
  export function forceCenter(x?: number, y?: number, z?: number): any;
  export function forceX(x?: number): any;
  export function forceY(y?: number): any;
  export function forceZ(z?: number): any;
  export default function forceSimulation(nodes?: any[]): any;
}
