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

  const header = page.locator("header");
  const primary = page.getByRole("navigation", { name: "Primary navigation" });
  await expect(primary.getByRole("link", { name: "Product" })).toBeVisible();
  await expect(primary.getByRole("link", { name: "How it works" })).toBeVisible();
  await expect(primary.getByRole("link", { name: "Pricing" })).toBeVisible();

  const contextual = page.getByRole("navigation", { name: "Basketball video analysis navigation" });
  await expect(contextual.getByRole("link", { name: "How it works" })).toBeVisible();
  await expect(contextual.getByRole("link", { name: "Capabilities" })).toBeVisible();

  const signIn = header.getByRole("link", { name: "Sign in", exact: true });
  const cta = header.getByRole("link", { name: "Analyze your first game free", exact: true });
  await expect(signIn).toBeVisible();
  await expect(cta).toBeVisible();

  for (const width of [320, 430, 820]) {
    await page.setViewportSize({ width, height: 844 });
    const menuButton = header.getByRole("button", { name: "Open navigation" });
    const language = header.locator("select[aria-label]");

    for (const control of [menuButton, language, signIn, cta]) {
      await expect(control).toBeVisible();
      const box = await control.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(width + 1);
    }
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await header.getByRole("button", { name: "Open navigation" }).click();
  const closeButton = header.getByRole("button", { name: "Close navigation" });
  await expect(closeButton).toHaveAttribute("aria-expanded", "true");

  const mobile = page.getByRole("navigation", { name: "Mobile primary navigation" });
  await expect(mobile.getByRole("link", { name: "Product" })).toBeVisible();

  await page.keyboard.press("Escape");
  const reopenedButton = header.getByRole("button", { name: "Open navigation" });
  await expect(reopenedButton).toHaveAttribute("aria-expanded", "false");
  await expect(reopenedButton).toBeFocused();
});

test("mobile menu closes after same-page anchor navigation", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  const header = page.locator("header");
  await header.getByRole("button", { name: "Open navigation" }).click();
  const mobile = page.getByRole("navigation", { name: "Mobile primary navigation" });
  await mobile.getByRole("link", { name: "How it works" }).click();

  await expect(page).toHaveURL(/\/#how$/);
  await expect(header.getByRole("button", { name: "Open navigation" })).toHaveAttribute("aria-expanded", "false");
  await expect(mobile).not.toBeVisible();
});

test("contextual anchors clear the sticky header at mobile breakpoints", async ({ page }) => {
  for (const width of [320, 430, 820]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto("/basketball-video-analysis");

    const header = page.locator("header");
    const contextual = page.getByRole("navigation", { name: "Basketball video analysis navigation" });
    await contextual.getByRole("link", { name: "Capabilities" }).click();
    await expect(page).toHaveURL(/#capabilities$/);

    const target = page.locator("#capabilities");
    await expect(target).toBeVisible();
    await expect.poll(async () => {
      const [headerBox, targetBox] = await Promise.all([header.boundingBox(), target.boundingBox()]);
      if (!headerBox || !targetBox) return -999;
      return Math.round(targetBox.y - (headerBox.y + headerBox.height));
    }).toBeGreaterThanOrEqual(-1);
  }
});

test("skip link moves focus to main content", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: /skip to main content|saltar para o conteúdo principal|saltar al contenido principal/i });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();
});
