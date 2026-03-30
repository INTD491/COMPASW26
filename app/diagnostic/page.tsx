"use client";

import { useEffect, useState } from "react";
import PlotlyChart from "@/components/PlotlyChart";
import {
  computeFloridaDiagnostics,
  computeGeorgiaDiagnostics,
  type DiagnosticResults,
  type HypothesisResult,
  type ChartDef,
  type Verdict,
} from "@/lib/diagnosticStats";
import type { Data, Layout } from "plotly.js";

type Tab = "florida" | "georgia";

// ── Verdict badge ──────────────────────────────────────────────────────────

function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const map: Record<Verdict, { bg: string; text: string; border: string }> = {
    "Supported":     { bg: "#dcfce7", text: "#166534", border: "#10b981" },
    "Not Supported": { bg: "#fee2e2", text: "#991b1b", border: "#ef4444" },
    "Inconclusive":  { bg: "#fef9c3", text: "#854d0e", border: "#f59e0b" },
  };
  const s = map[verdict];
  return (
    <span style={{
      background: s.bg, color: s.text,
      border: `1px solid ${s.border}`,
      borderRadius: "6px", padding: "3px 10px",
      fontSize: "12px", fontWeight: 700, whiteSpace: "nowrap",
    }}>
      {verdict}
    </span>
  );
}

// ── Chart renderer ─────────────────────────────────────────────────────────

function HypothesisChart({ def }: { def: ChartDef }) {
  const COLORS = ["#0ea5e9", "#ef4444", "#f59e0b", "#10b981", "#8b5cf6"];

  const data: Data[] = def.traces.map((t, i) => ({
    type: "bar",
    name: t.name,
    x: t.x,
    y: t.y,
    marker: { color: t.color ?? COLORS[i % COLORS.length] },
  }));

  const layout: Partial<Layout> = {
    title: { text: def.title, font: { size: 14, color: "#1e293b" } },
    barmode: def.traces.length > 1 ? "group" : "relative",
    xaxis: { title: { text: def.xLabel ?? "", font: { size: 12 } }, tickfont: { size: 11 } },
    yaxis: { title: { text: def.yLabel ?? "", font: { size: 12 } }, tickfont: { size: 11 } },
    legend: { font: { size: 11 }, orientation: "h", y: -0.2 },
    margin: { t: 50, b: 80, l: 60, r: 20 },
    plot_bgcolor: "#f8fafc",
    paper_bgcolor: "white",
    font: { family: "inherit" },
    height: 320,
  };

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      style={{ width: "100%", borderRadius: "8px", overflow: "hidden" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}

// ── Hypothesis card ────────────────────────────────────────────────────────

const hypothesisText: Record<string, { statement: string; rationale: string; implication: string }> = {
  H1: {
    statement: "Race acts as a proxy for unmeasured socioeconomic and geographic factors",
    rationale: "Neighbourhood policing intensity, access to legal representation, and socioeconomic opportunity are not in the dataset. Race correlates with these omitted variables, so the model partially learns socioeconomic disadvantage via the race feature.",
    implication: "Removing race alone will not eliminate bias — the signal leaks through correlated variables like priors_count and charge degree.",
  },
  H2: {
    statement: "The COMPAS algorithm applies internal score thresholds, not a continuous risk model",
    rationale: "A smooth continuous regression model would produce a roughly uniform or bell-shaped distribution of scores. Heavy clustering at the extremes (1–2 and 9–10) is the signature of a threshold-based or rule-based classification system.",
    implication: "Threshold-based models amplify small differences near cut-points, exaggerating disparities between groups that are similarly situated near a boundary.",
  },
  H3: {
    statement: "Zero-inflated juvenile counts cause disproportionate penalisation of those with any juvenile history",
    rationale: "When ~70% of values are zero, any non-zero count is a statistical outlier. Linear models assign a large coefficient to juvenile felony count because it predicts recidivism in the small non-zero group, but that coefficient then over-applies to marginal cases.",
    implication: "A zero-inflated or categorical treatment of juvenile history would produce fairer predictions than a raw continuous input.",
  },
  H4: {
    statement: "Prior crimes count encodes historical enforcement bias, not just individual behaviour",
    rationale: "Communities with higher policing density accumulate more arrests per resident regardless of actual offending rates. If African American defendants are policed more intensively, their prior crimes counts reflect enforcement disparities as much as behaviour.",
    implication: "Using prior crimes count as a neutral predictor imports historical enforcement inequities directly into the model.",
  },
  H5: {
    statement: "Employment post-release is the strongest structural stabiliser against recidivism",
    rationale: "Employment reduces recidivism through multiple mechanisms: income (reducing material need for crime), structure and routine, social bonds with coworkers, and supervision compliance incentives.",
    implication: "Interventions targeting employment stability may reduce recidivism more effectively than supervision intensity increases. Models that exclude employment underestimate the modifiable component of risk.",
  },
  H6: {
    statement: "The 3-year binary outcome conflates two distinct recidivism phases with different predictors",
    rationale: "Re-arrest risk is typically highest in the first 6–12 months post-release (early failure, often supervision violations) and then declines. Year 1 and Year 3 recidivism likely have different predictors.",
    implication: "A binary 3-year outcome obscures when recidivism occurs. Survival analysis would reveal whether the same variables predict early vs. late failure.",
  },
  H7: {
    statement: "The supervision risk score is partially self-fulfilling due to surveillance intensity",
    rationale: "A higher risk score triggers more intensive supervision, which creates more opportunities to detect violations. This feedback loop means high-scored individuals are more likely to be caught for the same behaviour.",
    implication: "If recidivism is measured as any re-arrest or supervision violation, the risk score's predictive validity is partly circular rather than reflecting genuine behavioural differences.",
  },
  H8: {
    statement: "Prison_Years is a proxy for offence severity, introducing an indirect confound",
    rationale: "Longer sentences are associated with more serious offences. Offence severity predicts both supervision conditions and post-release recidivism risk. Including Prison_Years as a continuous variable imports severity effects without explicitly modelling them.",
    implication: "Models should either directly include offence type/severity or treat Prison_Years categorically (short/medium/long) to avoid inadvertently encoding severity as a linear effect.",
  },
};

function HypothesisCard({ result }: { result: HypothesisResult }) {
  const text = hypothesisText[result.id];
  const [open, setOpen] = useState(true);
  const verdictBorderColor: Record<Verdict, string> = {
    "Supported":     "#10b981",
    "Not Supported": "#ef4444",
    "Inconclusive":  "#f59e0b",
  };

  return (
    <section className="section-card" style={{ borderTop: `4px solid ${verdictBorderColor[result.verdict]}` }}>
      <div className="section-card__body">

        {/* Header row */}
        <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", flexWrap: "wrap" }}>
          <span style={{
            background: "#0ea5e9", color: "white",
            borderRadius: "6px", padding: "3px 10px",
            fontSize: "13px", fontWeight: 700, flexShrink: 0,
          }}>
            {result.id}
          </span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 700, fontSize: "15px", color: "#1e293b", marginBottom: "4px" }}>
              {text?.statement}
            </div>
            <div style={{ fontSize: "12px", color: "#64748b" }}>{result.testName}</div>
          </div>
          <VerdictBadge verdict={result.verdict} />
        </div>

        {/* Test statistics */}
        <div style={{
          display: "flex", gap: "24px", flexWrap: "wrap",
          background: "#f8fafc", borderRadius: "8px", padding: "12px 16px",
          marginTop: "14px",
        }}>
          <div>
            <div style={{ fontSize: "11px", color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>{result.statLabel}</div>
            <div style={{ fontSize: "22px", fontWeight: 700, color: "#0ea5e9", marginTop: "2px" }}>{result.statValue}</div>
          </div>
          {result.pValue !== null && (
            <div>
              <div style={{ fontSize: "11px", color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>p-value</div>
              <div style={{ fontSize: "22px", fontWeight: 700, color: result.pValue < 0.001 ? "#10b981" : result.pValue < 0.05 ? "#f59e0b" : "#ef4444", marginTop: "2px" }}>
                {result.pValue < 0.001 ? "< 0.001" : result.pValue.toFixed(4)}
              </div>
            </div>
          )}
          <div style={{ flex: 1, minWidth: "200px" }}>
            <div style={{ fontSize: "11px", color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "4px" }}>Result summary</div>
            <p style={{ margin: 0, fontSize: "13px", color: "#374151", lineHeight: 1.5 }}>{result.summary}</p>
          </div>
        </div>

        {/* Charts */}
        <div style={{ display: "grid", gridTemplateColumns: result.charts.length > 1 ? "1fr 1fr" : "1fr", gap: "16px", marginTop: "16px" }}>
          {result.charts.map((c, i) => <HypothesisChart key={i} def={c} />)}
        </div>

        {/* Expandable rationale */}
        <button
          onClick={() => setOpen(o => !o)}
          style={{ background: "none", border: "none", cursor: "pointer", fontSize: "13px", color: "#0ea5e9", fontWeight: 600, padding: "10px 0 0 0", display: "flex", alignItems: "center", gap: "4px" }}
        >
          {open ? "▾" : "▸"} {open ? "Hide" : "Show"} rationale & fairness implication
        </button>
        {open && text && (
          <div style={{ marginTop: "8px", display: "grid", gap: "8px" }}>
            <div style={{ padding: "12px 14px", background: "#f0f9ff", borderRadius: "8px", borderLeft: "4px solid #0ea5e9", fontSize: "13px", color: "#0c4a6e" }}>
              <strong>Rationale:</strong> {text.rationale}
            </div>
            <div style={{ padding: "12px 14px", background: "#fef3c7", borderRadius: "8px", borderLeft: "4px solid #f59e0b", fontSize: "13px", color: "#78350f" }}>
              <strong>Fairness implication:</strong> {text.implication}
            </div>
          </div>
        )}

      </div>
    </section>
  );
}

// ── Summary bar ────────────────────────────────────────────────────────────

function SummaryBar({ hypotheses }: { hypotheses: HypothesisResult[] }) {
  const supported    = hypotheses.filter(h => h.verdict === "Supported").length;
  const inconclusive = hypotheses.filter(h => h.verdict === "Inconclusive").length;
  const notSupported = hypotheses.filter(h => h.verdict === "Not Supported").length;
  return (
    <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginBottom: "24px" }}>
      {[
        { label: "Supported", count: supported, bg: "#dcfce7", text: "#166534", border: "#10b981" },
        { label: "Inconclusive", count: inconclusive, bg: "#fef9c3", text: "#854d0e", border: "#f59e0b" },
        { label: "Not Supported", count: notSupported, bg: "#fee2e2", text: "#991b1b", border: "#ef4444" },
      ].map(s => (
        <div key={s.label} style={{
          background: s.bg, color: s.text, border: `1px solid ${s.border}`,
          borderRadius: "8px", padding: "8px 16px", display: "flex", alignItems: "center", gap: "8px",
        }}>
          <span style={{ fontSize: "22px", fontWeight: 700 }}>{s.count}</span>
          <span style={{ fontSize: "13px", fontWeight: 500 }}>{s.label}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function DiagnosticPage() {
  const [activeTab, setActiveTab] = useState<Tab>("florida");
  const [florida, setFlorida] = useState<DiagnosticResults | null>(null);
  const [georgia, setGeorgia] = useState<DiagnosticResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([computeFloridaDiagnostics(), computeGeorgiaDiagnostics()])
      .then(([f, g]) => { setFlorida(f); setGeorgia(g); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  const data = activeTab === "florida" ? florida : georgia;

  const tabStyle = (tab: Tab) => ({
    padding: "8px 20px", borderRadius: "6px", border: "none", cursor: "pointer",
    fontSize: "14px", fontWeight: 500,
    background: activeTab === tab ? "#0ea5e9" : "#f1f5f9",
    color: activeTab === tab ? "white" : "#374151",
    transition: "all 0.15s",
  });

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Diagnostic Analysis</p>
        <h1>Hypothesis Testing on Real Data</h1>
        <p>
          Each hypothesis is tested against the actual dataset. Statistical tests are computed
          client-side from the raw CSVs — every chart and verdict reflects the real data.
        </p>
      </section>

      <div style={{ display: "flex", gap: "10px", marginBottom: "24px" }}>
        <button style={tabStyle("florida")} onClick={() => setActiveTab("florida")}>Florida COMPAS</button>
        <button style={tabStyle("georgia")} onClick={() => setActiveTab("georgia")}>Georgia NIJ</button>
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: "60px", color: "#64748b" }}>
          Loading data and running statistical tests…
        </div>
      )}

      {error && (
        <div style={{ padding: "16px", background: "#fee2e2", borderRadius: "8px", color: "#991b1b", marginBottom: "24px" }}>
          Failed to load data: {error}
        </div>
      )}

      {!loading && data && (
        <>
          <SummaryBar hypotheses={data.hypotheses} />
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {data.hypotheses.map(h => <HypothesisCard key={h.id} result={h} />)}
          </div>
        </>
      )}
    </main>
  );
}
