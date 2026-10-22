import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',

  // Three.js and A-Frame come from CDN globals
  resolve: {
    alias: {
      'three': 'three',
    },
  },

  optimizeDeps: {
    exclude: ['three'],
  },

  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      external: ['three'],
      output: {
        globals: {
          three:  'THREE',
        },
      },
    },
  },

  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws':  { target: 'ws://127.0.0.1:8000',  ws: true },
    },
  },
});
