import { defineConfig } from "vite";

const proxy = {
  "/xr-frame": {
    target: "http://127.0.0.1:8766",
    rewrite: () => "/frame",
    proxyTimeout: 1500,
  },
};

export default defineConfig({
  server: { proxy },
  preview: { proxy },
  optimizeDeps: { exclude: ["pianocv"] },
  worker: { format: "es" },
});
