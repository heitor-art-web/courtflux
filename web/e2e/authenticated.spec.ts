import { expect, test, type Page } from "@playwright/test";

const email = process.env.COURTFLUX_E2E_EMAIL;
const password = process.env.COURTFLUX_E2E_PASSWORD;

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email").fill(email!);
  await page.getByLabel("Password").fill(password!);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/account(?:$|[?#])/);
}

test.describe("authenticated CourtFlux flows", () => {
  test.skip(!email || !password, "Set COURTFLUX_E2E_EMAIL and COURTFLUX_E2E_PASSWORD to run authenticated E2E checks.");

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test("dashboard, quota state, analyses, teams and account routes stay navigable", async ({ page }) => {
    await expect(page.getByRole("heading", { name: /welcome back/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /my analyses/i })).toBeVisible();

    const startAnalysis = page.getByRole("link", { name: /start new analysis/i });
    const unavailable = page.getByText("Analysis unavailable", { exact: true });
    expect((await startAnalysis.count()) + (await unavailable.count())).toBeGreaterThan(0);
    if (await unavailable.count()) await expect(unavailable.first()).toHaveAttribute("aria-disabled", "true");

    await page.goto("/analyses");
    await expect(page.getByRole("heading", { name: /analysis history/i })).toBeVisible();

    await page.goto("/teams");
    await expect(page.getByRole("heading", { name: /keep repeat team context/i })).toBeVisible();

    await page.goto("/account");
    await expect(page.getByRole("heading", { name: /welcome back/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /monthly usage/i })).toBeVisible();
  });

  test("New Analysis exposes accessible Upload and YouTube source state", async ({ page }) => {
    await page.goto("/analysis/new");
    const upload = page.getByRole("button", { name: "Upload video" });
    const youtube = page.getByRole("button", { name: "YouTube URL" });
    await expect(upload).toHaveAttribute("aria-pressed", "true");
    await expect(youtube).toHaveAttribute("aria-pressed", "false");

    await youtube.click();
    await expect(upload).toHaveAttribute("aria-pressed", "false");
    await expect(youtube).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator('input[type="url"], input[placeholder*="YouTube"]')).toBeVisible();

    await upload.click();
    const fileInput = page.locator("#video-file");
    await fileInput.focus();
    await expect(fileInput).toBeFocused();
  });

  test("language switching survives authenticated routes and localizes dates", async ({ page }) => {
    const language = page.getByRole("combobox", { name: /language|língua|idioma/i });
    await expect(language).toBeVisible();
    await language.selectOption("pt-PT");
    await expect(page.locator("html")).toHaveAttribute("lang", "pt-PT");

    await page.goto("/analyses");
    await expect(page.locator("html")).toHaveAttribute("lang", "pt-PT");
    await expect(page.locator("body")).not.toContainText(/\b(?:Jan|Feb|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b/);

    await page.getByRole("combobox", { name: /language|língua|idioma/i }).selectOption("es-ES");
    await expect(page.locator("html")).toHaveAttribute("lang", "es-ES");
  });

  test("opens Processing or Results when the fixture has an analysis", async ({ page }) => {
    await page.goto("/analyses");
    const analysisLinks = page.locator('a[href^="/analysis/"]');
    if ((await analysisLinks.count()) === 0) test.skip(true, "Fixture has no existing analysis.");
    await analysisLinks.first().click();
    await expect(page).toHaveURL(/\/analysis\//);
    await expect(page.getByText(/processing progress|analysis results|processing analysis/i).first()).toBeVisible();

    const tabs = page.getByRole("tab");
    if (await tabs.count()) {
      await expect(tabs.first()).toHaveAttribute("tabindex", "0");
      await tabs.first().focus();
      await page.keyboard.press("ArrowRight");
      await expect(tabs.nth(1)).toBeFocused();
      await expect(tabs.nth(1)).toHaveAttribute("aria-controls");
      await expect(page.getByRole("tabpanel")).toBeVisible();
    }
  });
});
