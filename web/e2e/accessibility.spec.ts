import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const paths = [
  "/",
  "/basketball-video-analysis",
  "/basketball-analysis-for-coaches",
  "/basketball-analysis-for-teams",
  "/trust",
  "/ai-limitations",
  "/privacy",
  "/terms",
  "/video-data",
  "/login",
] as const;

for (const path of paths) {
  test(`axe smoke: ${path}`, async ({ page }) => {
    await page.goto(path);
    await expect(page.locator("#main-content")).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(results.violations, JSON.stringify(results.violations, null, 2)).toEqual([]);
  });
}
