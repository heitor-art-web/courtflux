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

test("language selector switches public route to Portuguese and Spanish", async ({ page }) => {
  await page.goto("/");
  
  // Wait for the select element to be stable and available in the DOM
  const languageSelect = page.locator("select[aria-label*='Language'], select[aria-label*='Língua'], select[aria-label*='Idioma']");
  await languageSelect.waitFor({ state: "attached", timeout: 10000 });
  await page.waitForTimeout(500); // Additional buffer for portal rendering
  
  // Verify it's visible
  await expect(languageSelect).toBeVisible({ timeout: 5000 });

  // Change to Portuguese
  await languageSelect.selectOption("pt-PT");
  await expect(page).toHaveURL(/\/pt-pt(?:$|[?#])/);
  await expect(page.locator("html")).toHaveAttribute("lang", "pt-PT");

  // Change to Spanish
  await languageSelect.selectOption("es-ES");
  await expect(page).toHaveURL(/\/es-es(?:$|[?#])/);
  await expect(page.locator("html")).toHaveAttribute("lang", "es-ES");
});

test("skip link moves focus to main content", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: /skip to main content|saltar para o conteúdo principal|saltar al contenido principal/i });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();
});
