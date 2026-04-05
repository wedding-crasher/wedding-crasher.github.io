import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://wedding-crasher.github.io",
  base: process.env.BASE_PATH ?? "",
  integrations: [sitemap()],
  markdown: {
    shikiConfig: {
      theme: "github-dark"
    }
  }
});