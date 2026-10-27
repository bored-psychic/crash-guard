import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on :5173 talks to the FastAPI backend on :8000 (CORS is open).
// `npm run build` emits web/dist which FastAPI serves at / for `make demo`.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: { outDir: "dist" },
});
