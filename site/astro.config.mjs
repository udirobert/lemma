import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  output: "static",
  site: "https://lemma.famile.xyz",
  integrations: [sitemap({ filter: (page) => !page.includes("/waitlist") })],
});
