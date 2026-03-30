"use client";

import dynamic from "next/dynamic";
import type { CSSProperties, ComponentType } from "react";
import type { Config, Data, Layout } from "plotly.js";

type PlotlyChartProps = {
  className?: string;
  config?: Partial<Config>;
  data: Data[];
  layout?: Partial<Layout>;
  style?: CSSProperties;
  useResizeHandler?: boolean;
};

const Plot = dynamic(
  async () => {
    const [factoryModule, plotlyModule] = await Promise.all([
      import("react-plotly.js/factory"),
      import("plotly.js-dist-min")
    ]);

    return factoryModule.default(
      (plotlyModule as { default?: unknown }).default ?? plotlyModule
    ) as ComponentType<PlotlyChartProps>;
  },
  {
    ssr: false,
    loading: () => <div className="chart-loading">Loading chart...</div>
  }
) as ComponentType<PlotlyChartProps>;

export default function PlotlyChart(props: PlotlyChartProps) {
  return <Plot {...props} />;
}

