import { expect, test } from "@playwright/test";

test("Homepage → primary beta CTA opens signup with keyboard focus preserved", async ({ page }) => {
  await page.goto("/");
  const main = page.locator("#main-content");
  await expect(main).toBeVisible();
  await expect(page.locator("body")).toHaveAttribute("data-public-page", "true");

  const cta = page
    .locator('section[aria-labelledby="hero-title"]')
    .locator('a[href="/login?mode=signup"]');
  await expect(cta).toBeVisible();
  await cta.focus();
  await expect(cta).toBeFocused();
  await cta.click();

  await expect(page).toHaveURL(/\/login\?mode=signup/);
  await expect(page.locator("body")).toHaveAttribute("data-public-page", "false");
  await expect(page.getByRole("heading", { name: "Create account" })).toBeVisible();
  await expect(page.getByLabel("Full name")).toBeVisible();
  await expect(page.getByLabel("Email")).toBeVisible();
  await expect(page.getByLabel("Password")).toBeVisible();
});

test("language selector is embedded in the public header and switches locales", async ({ page }) => {
  await page.goto("/");
  const language = page.locator("header select[aria-label]");
  await expect(language).toHaveCount(1);
  await expect(language).toBeVisible();

  await language.selectOption("pt-PT");
  await expect(page).toHaveURL(/\/pt-pt(?:$|[?#])/);
  await expect(page.locator("html")).toHaveAttribute("lang", "pt-PT");
  await expect(page.locator("body")).toHaveAttribute("data-public-page", "true");

  const portugueseLanguage = page.locator("header select[aria-label]");
  await expect(portugueseLanguage).toHaveCount(1);
  await portugueseLanguage.selectOption("es-ES");
  await expect(page).toHaveURL(/\/es-es(?:$|[?#])/);
  await expect(page.locator("html")).toHaveAttribute("lang", "es-ES");
  await expect(page.locator("body")).toHaveAttribute("data-public-page", "true");
});

test("public navigation separates global, contextual and mobile actions", async ({ page }) => {
  await page.goto("/basketball-video-analysis");

  const primary = page.getByRole("navigation", { name: "Primary navigation" });
  await expect(primary.getByRole("link", { name: "Product" })).toBeVisible();
  await expect(primary.getByRole("link", { name: "How it works" })).toBeVisible();
  await expect(primary.getByRole("link", { name: "Pricing" })).toBeVisible();

  const contextual = page.getByRole("navigation", { name: "Basketball video analysis navigation" });
  await expect(contextual.getByRole("link", { name: "How it works" })).toBeVisible();
  await expect(contextual.getByRole("link", { name: "Capabilities" })).toBeVisible();

  await expect(page.getByRole("link", { name: "Sign in", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Analyze your first game free", exact: true })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("link", { name: "Sign in", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Analyze your first game free", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Open navigation" }).click().catch(async () => {
    await page.getByText("Menu", { exact: true }).click();
  });
  const mobile = page.getByRole("navigation", { name: "Mobile primary navigation" });
  await expect(mobile.getByRole("link", { name: "Product" })).toBeVisible();
});

test("skip link moves focus to main content", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: /skip to main content|saltar para o conteúdo principal|saltar al contenido principal/i });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();
});
