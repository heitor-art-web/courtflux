import { expect, test } from "@playwright/test";

test("homepage exposes primary beta CTA and keyboard focus", async ({ page }) => {
  await page.goto("/");
  const main = page.locator("#main-content");
  await expect(main).toBeVisible();

  const cta = page.getByRole("link", { name: /start|beta|analysis/i }).first();
  await expect(cta).toBeVisible();
  await cta.focus();
  await expect(cta).toBeFocused();
});

test("language selector switches public route to Portuguese and Spanish", async ({ page }) => {
  await page.goto("/");
  const language = page.getByRole("combobox", { name: /language|língua|idioma/i });
  await expect(language).toBeVisible();

  await language.selectOption("pt-PT");
  await expect(page).toHaveURL(/\/pt-pt(?:$|[?#])/);
  await expect(page.locator("html")).toHaveAttribute("lang", "pt-PT");

  await page.getByRole("combobox", { name: /language|língua|idioma/i }).selectOption("es-ES");
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
