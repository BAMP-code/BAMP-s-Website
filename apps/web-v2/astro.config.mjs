import { defineConfig } from "astro/config";

export default defineConfig({
  // Lock the dev server to 4321 so it doesn't collide with apps/web (Next on 3000).
  server: { port: 4321 },
});
