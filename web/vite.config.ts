import { defineConfig } from 'vite';

// three.js is bundled (not a CDN global) so the mind view can use
// three/examples/jsm (OrbitControls, EffectComposer, UnrealBloomPass).
// A-Frame, which shipped its own copy of three, has been removed.
export default defineConfig({
  base: '/',

  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 1200,
  },

  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws':  { target: 'ws://127.0.0.1:8000',  ws: true },
    },
  },
});
