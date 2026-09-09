import { expect, test, type Page } from "@playwright/test";

const email = process.env.COURTFLUX_E2E_ONBOARDING_EMAIL;
const password = process.env.COURTFLUX_E2E_ONBOARDING_PASSWORD;

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email").fill(email!);
  await page.getByLabel("Password").fill(password!);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForLoadState("networkidle");
}

test("onboarding role radiogroup supports keyboard navigation", async ({ page }) => {
  test.skip(!email || !password, "Set dedicated onboarding E2E credentials to run this fixture test.");
  await login(page);
  await page.goto("/onboarding");

  const group = page.getByRole("radiogroup", { name: "Basketball role" });
  await expect(group).toBeVisible();
  const radios = group.getByRole("radio");
  await expect(radios.first()).toHaveAttribute("tabindex", /0|-1/);

  const initiallyFocusable = radios.filter({ has: page.locator('[tabindex="0"]') });
  if (await initiallyFocusable.count()) await initiallyFocusable.first().focus();
  else await radios.first().focus();

  await page.keyboard.press("ArrowDown");
  await expect(radios.nth(1)).toBeFocused();
  await expect(radios.nth(1)).toHaveAttribute("aria-checked", "true");
});
