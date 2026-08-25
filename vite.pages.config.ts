import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/philpapers-atlas/',
  plugins: [react()],
  publicDir: 'public',
  build: {
    outDir: 'pages-dist',
    emptyOutDir: true,
    sourcemap: false,
    target: 'es2020',
  },
});
