import { render } from "@testing-library/react";

const plotlySpy = vi.fn(() => <div data-testid="plotly-chart" />);

vi.mock("@/components/PlotlyChart", () => ({
  default: (props: unknown) => plotlySpy(props)
}));

import { ClassDistributionChart } from "@/components/ClassDistributionChart";

describe("ClassDistributionChart", () => {
  it("passes ten bars to Plotly", () => {
    render(
      <ClassDistributionChart
        counts={[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]}
        labels={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
      />
    );

    const props = plotlySpy.mock.calls[0][0] as {
      data: Array<{ x: number[]; y: number[] }>;
    };

    expect(props.data[0].x).toHaveLength(10);
    expect(props.data[0].y).toHaveLength(10);
  });
});
