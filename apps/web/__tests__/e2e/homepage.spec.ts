import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.describe("Homepage", () => {
  test("should load without console errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    await page.goto("/");
    await expect(page).toHaveTitle(/BAMP/);
    expect(errors).toHaveLength(0);
  });

  test("should have all main sections visible", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#main-content")).toBeVisible();
    await expect(page.getByText("ABOUT")).toBeVisible();
    await expect(page.getByText("Computer Science Projects")).toBeVisible();
    await expect(page.getByText("EE and ME Projects")).toBeVisible();
    await expect(page.getByText("Drawings")).toBeVisible();
  });

  test("should pass axe accessibility scan", async ({ page }) => {
    await page.goto("/");
    const results = await new AxeBuilder({ page }).analyze();
    expect(results.violations).toEqual([]);
  });

  test("slider should have carousel role", async ({ page }) => {
    await page.goto("/");
    const carousels = page.locator('[aria-roledescription="carousel"]');
    await expect(carousels).toHaveCount(3);
  });

  test("chatbot launcher should be visible", async ({ page }) => {
    await page.goto("/");
    const launcher = page.getByLabel(/movie recommendation chatbot/i);
    await expect(launcher).toBeVisible();
  });
});
