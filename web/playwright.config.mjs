import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "test", testMatch: /.*\.spec\.mjs/, timeout: 15 * 60_000, workers: 1, reporter: [["list"]],
  use: { baseURL: "http://127.0.0.1:8787" },
  webServer: { command: "node server.mjs", url: "http://127.0.0.1:8787/", reuseExistingServer: true },
});
