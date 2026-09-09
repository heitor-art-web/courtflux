import { expect, test } from "@playwright/test";

const email = process.env.COURTFLUX_E2E_EMAIL;
const password = process.env.COURTFLUX_E2E_PASSWORD;

async function login(page: Parameters<typeof test>[0] extends never ? never : any) {
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

  test("dashboard, analyses, teams and account routes stay navigable", async ({ page }) => {
    await expect(page.getByRole("heading", { name: /welcome back/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /my analyses/i })).toBeVisible();
    await expect(page.getByText(/start new analysis|analysis unavailable/i)).toBeVisible();

    await page.goto("/analyses");
    await expect(page.getByRole("heading", { name: /analysis history/i })).toBeVisible();

    await page.goto("/teams");
    await expect(page.getByRole("heading", { name: /keep repeat team context/i })).toBeVisible();

    await page.goto("/account");
    await expect(page.getByRole("heading", { name: /welcome back/i })).toBeVisible();
  });

  test("New Analysis exposes accessible source state and upload focus", async ({ page }) => {
    await page.goto("/analysis/new");
    const upload = page.getByRole("button", { name: "Upload video" });
    const youtube = page.getByRole("button", { name: "YouTube URL" });
    await expect(upload).toHaveAttribute("aria-pressed", "true");
    await expect(youtube).toHaveAttribute("aria-pressed", "false");

    await youtube.click();
    await expect(upload).toHaveAttribute("aria-pressed", "false");
    await expect(youtube).toHaveAttribute("aria-pressed", "true");

    await upload.click();
    const fileInput = page.locator("#video-file");
    await fileInput.focus();
    await expect(fileInput).toBeFocused();
  });

  test("language switching survives authenticated routes", async ({ page }) => {
    const language = page.getByRole("combobox", { name: /language|língua|idioma/i });
    await expect(language).toBeVisible();
    await language.selectOption("pt-PT");
    await expect(page.locator("html")).toHaveAttribute("lang", "pt-PT");

    await page.goto("/analyses");
    await expect(page.locator("html")).toHaveAttribute("lang", "pt-PT");

    await page.getByRole("combobox", { name: /language|língua|idioma/i }).selectOption("es-ES");
    await expect(page.locator("html")).toHaveAttribute("lang", "es-ES");
  });

  test("opens an existing analysis when the fixture has one", async ({ page }) => {
    await page.goto("/analyses");
    const analysisLinks = page.locator('a[href^="/analysis/"]');
    if ((await analysisLinks.count()) === 0) test.skip(true, "Fixture has no existing analysis.");
    await analysisLinks.first().click();
    await expect(page).toHaveURL(/\/analysis\//);
    await expect(page.getByText(/processing progress|analysis results|processing analysis/i).first()).toBeVisible();
  });
});
