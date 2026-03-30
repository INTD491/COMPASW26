import { render, screen } from "@testing-library/react";

vi.mock("@/components/ClassDistributionChart", () => ({
  ClassDistributionChart: () => <div>Distribution Chart</div>
}));

vi.mock("@/components/DrawingCanvas", () => ({
  DrawingCanvas: () => <div>Drawing Canvas</div>
}));

import HomePage from "@/app/page";

describe("HomePage", () => {
  it("renders the main sections from the generated stats file", async () => {
    const page = await HomePage();
    render(page);

    expect(screen.getByRole("heading", { name: "MNIST Dataset Snapshot" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Class Distribution" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Draw a Digit" })).toBeInTheDocument();
    expect(screen.getByText(/public\/data\/mnist_stats\.json/i)).toBeInTheDocument();
  });
});

