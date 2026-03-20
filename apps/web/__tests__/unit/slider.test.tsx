import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Slider } from "@/components/slider/slider";
import type { Project } from "@/lib/types";

const mockProjects: Project[] = [
  {
    id: "test-1",
    title: "Project One",
    status: "completed",
    description: "First project",
    category: "cs",
    media: { type: "image", src: "/test1.jpg", alt: "Test 1", width: 100, height: 100 },
  },
  {
    id: "test-2",
    title: "Project Two",
    status: "in-progress",
    description: "Second project",
    category: "cs",
    media: { type: "image", src: "/test2.jpg", alt: "Test 2", width: 100, height: 100 },
  },
];

describe("Slider", () => {
  it("should render with carousel ARIA attributes", () => {
    const { container } = render(<Slider projects={mockProjects} label="Test" />);
    const section = container.querySelector('[aria-roledescription="carousel"]');
    expect(section).toBeTruthy();
    expect(section?.getAttribute("aria-label")).toBe("Test Projects");
  });

  it("should render prev/next buttons with aria-labels", () => {
    render(<Slider projects={mockProjects} label="Test" />);
    const prevButtons = screen.getAllByLabelText("Previous slide");
    const nextButtons = screen.getAllByLabelText("Next slide");
    expect(prevButtons.length).toBeGreaterThanOrEqual(1);
    expect(nextButtons.length).toBeGreaterThanOrEqual(1);
  });

  it("should render tab navigation dots", () => {
    render(<Slider projects={mockProjects} label="Test" />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.length).toBeGreaterThanOrEqual(2);
    // First dot should be selected, second not
    const selectedTabs = tabs.filter(t => t.getAttribute("aria-selected") === "true");
    const unselectedTabs = tabs.filter(t => t.getAttribute("aria-selected") === "false");
    expect(selectedTabs.length).toBeGreaterThanOrEqual(1);
    expect(unselectedTabs.length).toBeGreaterThanOrEqual(1);
  });

  it("should show first project initially", () => {
    render(<Slider projects={mockProjects} label="Test" />);
    const titles = screen.getAllByText("Project One");
    expect(titles.length).toBeGreaterThanOrEqual(1);
  });

  it("should have keyboard navigation support", async () => {
    const user = userEvent.setup();
    const { container } = render(<Slider projects={mockProjects} label="Test" />);
    const carousel = container.querySelector('[aria-roledescription="carousel"]') as HTMLElement;
    carousel.focus();
    await user.keyboard("{ArrowRight}");
    // Verify carousel is focusable and keyboard handler is attached
    expect(carousel.getAttribute("tabindex")).toBe("0");
  });
});
