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
  const focusable = group.locator('[role="radio"][tabindex="0"]');
  await expect(focusable).toHaveCount(1);
  await focusable.focus();

  const startIndex = await radios.evaluateAll((nodes) => nodes.findIndex((node) => node === document.activeElement));
  await page.keyboard.press("ArrowDown");
  const expectedIndex = (startIndex + 1) % (await radios.count());
  await expect(radios.nth(expectedIndex)).toBeFocused();
  await expect(radios.nth(expectedIndex)).toHaveAttribute("aria-checked", "true");
});
