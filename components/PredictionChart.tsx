"use client";

import PlotlyChart from "@/components/PlotlyChart";

type PredictionChartProps = {
  probabilities?: number[] | null;
  counts?: number[];
  labels?: string[];
};

export function PredictionChart({ probabilities, counts, labels }: PredictionChartProps) {
  const data = probabilities || counts;
  const chartLabels = labels || Array.from({ length: 10 }, (_, index) => index.toString());
  
  if (!data) {
    return (
      <div className="chart-placeholder">
        {probabilities === undefined 
          ? "Model accuracy comparison data not available."
          : "Run a prediction to see the class probabilities for digits 0 through 9."}
      </div>
    );
  }

  const isPercentage = counts ? Math.max(...counts) > 1 : false;

  return (
    <PlotlyChart
      className="chart-frame"
      config={{ displayModeBar: false, responsive: true }}
      data={[
        {
          type: "bar",
          y: chartLabels,
          x: data,
          orientation: "h",
          marker: {
            color: data.map((value, index) =>
              value === Math.max(...data)
                ? "#10b981"
                : index % 2 === 0
                  ? "#0ea5e9"
                  : "#06b6d4"
            )
          },
          hovertemplate: "%{y}<br>%{x:.1f}%<extra></extra>"
        }
      ]}
      layout={{
        autosize: true,
        height: 320,
        margin: { t: 12, l: 120, r: 18, b: 44 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {
          title: { text: counts ? "Accuracy (%)" : "Probability" },
          gridcolor: "rgba(30, 35, 48, 0.12)",
          range: [0, Math.max(...data) * 1.1]
        },
        yaxis: {
          automargin: true
        }
      }}
      style={{ width: "100%", height: "320px" }}
      useResizeHandler
    />
  );
}

