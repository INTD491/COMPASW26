"use client";

import PlotlyChart from "@/components/PlotlyChart";

type ClassDistributionChartProps = {
  counts: number[];
  labels: string[];
};

export function ClassDistributionChart({
  counts,
  labels
}: ClassDistributionChartProps) {
  return (
    <PlotlyChart
      className="chart-frame"
      config={{ displayModeBar: false, responsive: true }}
      data={[
        {
          type: "bar",
          x: labels,
          y: counts,
          marker: {
            color: "#0ea5e9",
            line: {
              color: "#1e3a5f",
              width: 1
            }
          },
          hovertemplate: "%{x}<br>Count: %{y}<extra></extra>"
        }
      ]}
      layout={{
        autosize: true,
        height: 320,
        margin: { t: 12, l: 48, r: 18, b: 80 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {
          title: { text: "Demographic Group" },
          tickmode: "array",
          tickvals: labels,
          tickangle: -45
        },
        yaxis: {
          title: { text: "Population Count" },
          gridcolor: "rgba(30, 35, 48, 0.12)"
        }
      }}
      style={{ width: "100%", height: "320px" }}
      useResizeHandler
    />
  );
}

