"use client";

import { useEffect, useState } from "react";
import {
  computeFloridaEDA,
  computeGeorgiaEDA,
  type EDAResult,
  type NumericStat,
  type CategoricalStat,
  type CorrelationStat,
} from "@/lib/edaStats";

// ── Numeric stats table ────────────────────────────────────────────────────

function NumericTable({ stats }: { stats: NumericStat[] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
        <thead>
          <tr style={{ borderBottom: "2px solid #e2e8f0", backgroundColor: "#f8fafc" }}>
            {["Variable", "n", "Mean", "Median", "Std Dev", "Min", "Max", "Q1", "Q3", "Skew", "Outliers"].map(
              (h) => (
                <th
                  key={h}
                  style={{
                    padding: "10px 12px",
                    textAlign: h === "Variable" ? "left" : "right",
                    fontWeight: 600,
                    color: "#374151",
                    whiteSpace: "nowrap",
                  }}
                >
                  {h}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {stats.map((s, i) => (
            <tr
              key={s.variable}
              style={{
                borderBottom: "1px solid #e2e8f0",
                backgroundColor: i % 2 === 0 ? "white" : "#f8fafc",
              }}
            >
              <td style={{ padding: "9px 12px", fontWeight: 500, color: "#1e293b" }}>{s.label}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.count.toLocaleString()}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.mean}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.median}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.std}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.min}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.max}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.q1}</td>
              <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>{s.q3}</td>
              <td
                style={{
                  padding: "9px 12px",
                  textAlign: "right",
                  color:
                    Math.abs(s.skewness) >= 0.5
                      ? "#ef4444"
                      : Math.abs(s.skewness) >= 0.2
                      ? "#f59e0b"
                      : "#10b981",
                  fontWeight: 500,
                }}
              >
                {s.skewness > 0 ? `+${s.skewness}` : s.skewness}
              </td>
              <td
                style={{
                  padding: "9px 12px",
                  textAlign: "right",
                  color: s.outliers > 0 ? "#f59e0b" : "#64748b",
                  fontWeight: s.outliers > 0 ? 500 : 400,
                }}
              >
                {s.outliers.toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Categorical distribution bars ─────────────────────────────────────────

function CategoricalCard({ stat }: { stat: CategoricalStat }) {
  const maxPct = stat.categories[0]?.pct ?? 1;
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e2e8f0",
        borderRadius: "10px",
        padding: "16px",
      }}
    >
      <div style={{ fontWeight: 600, fontSize: "14px", color: "#1e293b", marginBottom: "12px" }}>
        {stat.label}
        <span style={{ fontWeight: 400, color: "#64748b", marginLeft: "8px", fontSize: "12px" }}>
          ({stat.total.toLocaleString()} records)
        </span>
      </div>
      {stat.categories.map((cat) => (
        <div key={cat.label} style={{ marginBottom: "7px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "12px",
              color: "#374151",
              marginBottom: "3px",
            }}
          >
            <span>{cat.label}</span>
            <span style={{ color: "#64748b" }}>
              {cat.count.toLocaleString()} ({cat.pct}%)
            </span>
          </div>
          <div
            style={{
              background: "#f1f5f9",
              borderRadius: "4px",
              height: "8px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                background: "#0ea5e9",
                height: "100%",
                borderRadius: "4px",
                width: `${(cat.pct / maxPct) * 100}%`,
                transition: "width 0.3s ease",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Correlation table ──────────────────────────────────────────────────────

function CorrelationTable({ correlations }: { correlations: CorrelationStat[] }) {
  const color = (c: CorrelationStat) => {
    if (c.direction === "none") return "#94a3b8";
    if (c.strength === "strong") return c.direction === "positive" ? "#0ea5e9" : "#ef4444";
    if (c.strength === "moderate") return c.direction === "positive" ? "#38bdf8" : "#f87171";
    return "#94a3b8";
  };

  const badge = (c: CorrelationStat) => {
    const s = c.strength.charAt(0).toUpperCase() + c.strength.slice(1);
    const bg =
      c.strength === "strong"
        ? "#dbeafe"
        : c.strength === "moderate"
        ? "#fef9c3"
        : "#f1f5f9";
    const text =
      c.strength === "strong"
        ? "#1d4ed8"
        : c.strength === "moderate"
        ? "#854d0e"
        : "#64748b";
    return (
      <span
        style={{
          background: bg,
          color: text,
          borderRadius: "4px",
          padding: "2px 7px",
          fontSize: "11px",
          fontWeight: 600,
        }}
      >
        {s}
      </span>
    );
  };

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
        <thead>
          <tr style={{ borderBottom: "2px solid #e2e8f0", backgroundColor: "#f8fafc" }}>
            {["Variable A", "Variable B", "r", "Direction", "Strength"].map((h) => (
              <th
                key={h}
                style={{
                  padding: "10px 12px",
                  textAlign: h === "r" ? "center" : "left",
                  fontWeight: 600,
                  color: "#374151",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {correlations.map((c, i) => (
            <tr
              key={i}
              style={{
                borderBottom: "1px solid #e2e8f0",
                backgroundColor: i % 2 === 0 ? "white" : "#f8fafc",
              }}
            >
              <td style={{ padding: "9px 12px", color: "#1e293b", fontWeight: 500 }}>{c.labelA}</td>
              <td style={{ padding: "9px 12px", color: "#1e293b", fontWeight: 500 }}>{c.labelB}</td>
              <td
                style={{
                  padding: "9px 12px",
                  textAlign: "center",
                  fontWeight: 700,
                  fontSize: "14px",
                  color: color(c),
                }}
              >
                {c.r > 0 ? `+${c.r}` : c.r}
              </td>
              <td style={{ padding: "9px 12px", color: "#475569", textTransform: "capitalize" }}>
                {c.direction}
              </td>
              <td style={{ padding: "9px 12px" }}>{badge(c)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Overview stat cards ────────────────────────────────────────────────────

function OverviewCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <p style={{ fontSize: "12px", color: "#64748b", margin: "4px 0 0 0" }}>{sub}</p>}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function EDAPage() {
  const [activeTab, setActiveTab] = useState<"florida" | "georgia">("florida");
  const [florida, setFlorida] = useState<EDAResult | null>(null);
  const [georgia, setGeorgia] = useState<EDAResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([computeFloridaEDA(), computeGeorgiaEDA()])
      .then(([f, g]) => {
        setFlorida(f);
        setGeorgia(g);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  const data = activeTab === "florida" ? florida : georgia;

  const tabStyle = (tab: "florida" | "georgia") => ({
    padding: "8px 20px",
    borderRadius: "6px",
    border: "none",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: 500,
    background: activeTab === tab ? "#0ea5e9" : "#f1f5f9",
    color: activeTab === tab ? "white" : "#374151",
    transition: "all 0.15s",
  });

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Exploratory Data Analysis</p>
        <h1>Descriptive Statistics</h1>
        <p>
          A complete statistical summary of both datasets, Florida and Georgia — variable distributions, central tendency,
          spread, missing data, outliers, and correlations between key features.
        </p>
      </section>

      {/* Tab selector */}
      <div style={{ display: "flex", gap: "10px", marginBottom: "24px" }}>
        <button style={tabStyle("florida")} onClick={() => setActiveTab("florida")}>
          Florida COMPAS
        </button>
        <button style={tabStyle("georgia")} onClick={() => setActiveTab("georgia")}>
          Georgia NIJ
        </button>
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: "60px", color: "#64748b" }}>
          Loading and computing statistics…
        </div>
      )}

      {error && (
        <div
          style={{
            padding: "16px",
            background: "#fee2e2",
            borderRadius: "8px",
            color: "#991b1b",
            marginBottom: "24px",
          }}
        >
          Failed to load data: {error}
        </div>
      )}

      {!loading && data && (
        <div className="section-grid">

          {/* Overview */}
          <section className="section-card">
            <div className="section-card__body">
              <div className="section-card__header">
                <h2>Dataset Overview — {data.datasetName}</h2>
                <p>High-level snapshot of the dataset</p>
              </div>
              <div className="metrics-grid">
                <OverviewCard
                  label="Total Records"
                  value={data.totalRecords.toLocaleString()}
                  sub="rows in dataset"
                />
                <OverviewCard
                  label="Total Variables"
                  value={String(data.totalColumns)}
                  sub="columns in CSV"
                />
                <OverviewCard
                  label="Recidivism Rate"
                  value={`${(data.recidivismRate * 100).toFixed(1)}%`}
                  sub={activeTab === "florida" ? "2-year re-arrest" : "3-year re-arrest"}
                />
                <OverviewCard
                  label="Missing Data"
                  value={data.missingData.length === 0 ? "None" : `${data.missingData.length} cols`}
                  sub={
                    data.missingData.length === 0
                      ? "No missing values found"
                      : "columns with missing values"
                  }
                />
              </div>
            </div>
          </section>

          {/* Numeric summary */}
          <section className="section-card">
            <div className="section-card__body">
              <div className="section-card__header">
                <h2>Numeric Variable Summary</h2>
                <p>
                  Central tendency, spread, and shape for all numeric features. Skew is
                  Pearson&apos;s skewness (mean−median)/std — green &lt;0.2, amber &lt;0.5, red ≥0.5.
                  Outliers counted via 1.5×IQR rule.
                </p>
              </div>
              <NumericTable stats={data.numericStats} />
            </div>
          </section>

          {/* Categorical distributions */}
          <section className="section-card">
            <div className="section-card__body">
              <div className="section-card__header">
                <h2>Categorical Distributions</h2>
                <p>Frequency of each category; top 10 shown per variable</p>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                  gap: "16px",
                  marginTop: "16px",
                }}
              >
                {data.categoricalStats.map((s) => (
                  <CategoricalCard key={s.variable} stat={s} />
                ))}
              </div>
            </div>
          </section>

          {/* Missing data */}
          <section className="section-card">
            <div className="section-card__body">
              <div className="section-card__header">
                <h2>Missing Data</h2>
                <p>Columns with at least one missing or empty value</p>
              </div>
              {data.missingData.length === 0 ? (
                <div
                  style={{
                    padding: "16px",
                    background: "#dcfce7",
                    borderRadius: "8px",
                    borderLeft: "4px solid #10b981",
                    color: "#166534",
                    fontWeight: 500,
                  }}
                >
                  No missing values detected across all tracked columns.
                </div>
              ) : (
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
                    <thead>
                      <tr style={{ borderBottom: "2px solid #e2e8f0", backgroundColor: "#f8fafc" }}>
                        {["Variable", "Missing Count", "Missing %", "Severity"].map((h) => (
                          <th
                            key={h}
                            style={{
                              padding: "10px 12px",
                              textAlign: h === "Variable" ? "left" : "right",
                              fontWeight: 600,
                              color: "#374151",
                            }}
                          >
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {data.missingData.map((m, i) => {
                        const severity =
                          m.pct >= 20 ? "High" : m.pct >= 5 ? "Moderate" : "Low";
                        const sColor =
                          severity === "High"
                            ? "#ef4444"
                            : severity === "Moderate"
                            ? "#f59e0b"
                            : "#10b981";
                        return (
                          <tr
                            key={m.variable}
                            style={{
                              borderBottom: "1px solid #e2e8f0",
                              backgroundColor: i % 2 === 0 ? "white" : "#f8fafc",
                            }}
                          >
                            <td style={{ padding: "9px 12px", color: "#1e293b", fontWeight: 500 }}>
                              {m.variable}
                            </td>
                            <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>
                              {m.missing.toLocaleString()}
                            </td>
                            <td style={{ padding: "9px 12px", textAlign: "right", color: "#475569" }}>
                              {m.pct}%
                            </td>
                            <td style={{ padding: "9px 12px", textAlign: "right" }}>
                              <span
                                style={{
                                  color: sColor,
                                  fontWeight: 600,
                                  fontSize: "12px",
                                }}
                              >
                                {severity}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>

          {/* Correlations */}
          <section className="section-card">
            <div className="section-card__body">
              <div className="section-card__header">
                <h2>Variable Correlations</h2>
                <p>
                  Pearson correlation coefficients between key features.
                  |r| ≥ 0.5 = strong, 0.3–0.5 = moderate, 0.1–0.3 = weak, &lt;0.1 = negligible.
                </p>
              </div>
              <CorrelationTable correlations={data.correlations} />

              <div
                style={{
                  marginTop: "16px",
                  padding: "14px 16px",
                  background: "#f0f9ff",
                  borderRadius: "8px",
                  borderLeft: "4px solid #0ea5e9",
                  fontSize: "13px",
                  color: "#0c4a6e",
                }}
              >
                {activeTab === "florida" ? (
                  <>
                    <strong>Key takeaway:</strong> The COMPAS decile score shows the strongest correlation
                    with 2-year recidivism, but the relationship is far from perfect — suggesting the
                    score captures signal yet introduces noise. Prior crimes count is also positively
                    associated with recidivism, while age shows a negative association (older individuals
                    recidivate at lower rates).
                  </>
                ) : (
                  <>
                    <strong>Key takeaway:</strong> The supervision risk score shows positive correlation
                    with both delinquency reports and recidivism. Employment (% days employed) is
                    negatively correlated with recidivism — higher employment is associated with lower
                    re-arrest rates. Prior felony arrests strongly predict prior felony convictions,
                    as expected.
                  </>
                )}
              </div>
            </div>
          </section>

          {/* Key patterns */}
          <section className="section-card">
            <div className="section-card__body">
              <div className="section-card__header">
                <h2>Notable Patterns &amp; Concerns</h2>
              </div>
              <div style={{ display: "grid", gap: "12px", marginTop: "16px" }}>
                {activeTab === "florida" ? (
                  <>
                    <FindingBlock color="#fef3c7" border="#f59e0b" title="Right-skewed prior crimes">
                      Prior crimes count and juvenile charge counts are heavily right-skewed — most
                      defendants have zero or few prior crimes, but a small group has very high counts
                      (&gt;20). These drive outlier counts and inflate the mean above the median.
                    </FindingBlock>
                    <FindingBlock color="#fee2e2" border="#ef4444" title="Decile score compression">
                      Decile scores cluster at the extremes (1–2 and 8–10), with fewer cases in the
                      middle range. This bimodal shape suggests the score separates cases into low/high
                      risk rather than a smooth continuum.
                    </FindingBlock>
                    <FindingBlock color="#dcfce7" border="#10b981" title="Age distribution">
                      The majority of defendants are between 26–45 years old. Recidivism rates decline
                      with age — the 18–25 cohort recidivates most frequently (≈53%), while the 56+
                      cohort recidivates least (≈18%).
                    </FindingBlock>
                    <FindingBlock color="#dbeafe" border="#0ea5e9" title="Racial composition">
                      African Americans make up the largest racial group in the dataset (≈51%), followed
                      by Caucasians (≈35%). This imbalance is important context for fairness analysis.
                    </FindingBlock>
                  </>
                ) : (
                  <>
                    <FindingBlock color="#fef3c7" border="#f59e0b" title="Employment as a protective factor">
                      Percent days employed shows substantial variation (mean ~40%) and a negative
                      correlation with recidivism. Individuals who are employed more of the time
                      post-release have meaningfully lower re-arrest rates.
                    </FindingBlock>
                    <FindingBlock color="#fee2e2" border="#ef4444" title="Supervision risk score skew">
                      The supervision risk score is right-skewed, with a long tail of high-risk
                      individuals. The IQR-based outlier count is notable — verify whether extreme
                      scores represent genuinely high-risk cases or data entry anomalies.
                    </FindingBlock>
                    <FindingBlock color="#dcfce7" border="#10b981" title="3-year recidivism window">
                      At ~50% recidivism within 3 years, the Georgia dataset has a higher base rate
                      than Florida&apos;s 2-year rate (~45%). Comparing these datasets directly requires
                      adjusting for the different follow-up windows.
                    </FindingBlock>
                    <FindingBlock color="#dbeafe" border="#0ea5e9" title="Prior arrest diversity">
                      The dataset captures arrest history across 5 offense types (felony, misdemeanor,
                      violent, property, drug). Most individuals have low counts in each category, but
                      the drug and property arrest columns show the broadest distributions.
                    </FindingBlock>
                  </>
                )}
              </div>
            </div>
          </section>

        </div>
      )}
    </main>
  );
}

// ── Small inline component ─────────────────────────────────────────────────

function FindingBlock({
  color,
  border,
  title,
  children,
}: {
  color: string;
  border: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        padding: "14px 16px",
        backgroundColor: color,
        borderRadius: "8px",
        borderLeft: `4px solid ${border}`,
      }}
    >
      <div style={{ fontWeight: 600, color: "#1e293b", marginBottom: "4px" }}>{title}</div>
      <p style={{ margin: 0, fontSize: "13px", color: "#475569" }}>{children}</p>
    </div>
  );
}
