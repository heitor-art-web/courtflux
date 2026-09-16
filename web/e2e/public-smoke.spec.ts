import { expect, test } from "@playwright/test";

test("Homepage → primary beta CTA opens signup with keyboard focus preserved", async ({ page }) => {
  await page.goto("/");
  const main = page.locator("#main-content");
  await expect(main).toBeVisible();

  const cta = page
    .locator('section[aria-labelledby="hero-title"]')
    .locator('a[href="/login?mode=signup"]');
  await expect(cta).toBeVisible();
  await cta.focus();
  await expect(cta).toBeFocused();
  await cta.click();

  await expect(page).toHaveURL(/\/login\?mode=signup/);
  await expect(page.getByRole("heading", { name: "Create account" })).toBeVisible();
  await expect(page.getByLabel("Full name")).toBeVisible();
  await expect(page.getByLabel("Email")).toBeVisible();
  await expect(page.getByLabel("Password")).toBeVisible();
});

test("language selector is present on homepage", async ({ page }) => {
  await page.goto("/");
  
  // Wait for main content to be visible first
  await expect(page.locator("#main-content")).toBeVisible();
  
  // Wait longer for the portal to render the language selector
  await page.waitForTimeout(1000);
  
  // Find any select element with aria-label attribute
  // The language switcher can have aria-label of "Language", "Língua", or "Idioma"
  const languageSelect = page.locator("select[aria-label]").first();
  
  // Verify it exists and is visible
  await expect(languageSelect).toBeVisible({ timeout: 5000 });
});

test("skip link moves focus to main content", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: /skip to main content|saltar para o conteúdo principal|saltar al contenido principal/i });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();
});
