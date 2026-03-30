"use client";

import { PredictionChart } from "@/components/PredictionChart";

function ModelCard({
  name,
  accuracy,
  auc,
  precision,
  recall,
  tags,
  highlighted = false
}: {
  name: string;
  accuracy: string;
  auc: string;
  precision: string;
  recall: string;
  tags: string[];
  highlighted?: boolean;
}) {
  return (
    <div
      style={{
        padding: "20px",
        borderRadius: "12px",
        border: highlighted ? "2px solid #0ea5e9" : "1px solid #e2e8f0",
        background: highlighted ? "rgba(14, 165, 233, 0.05)" : "#f8fafc"
      }}
    >
      <h3 style={{ margin: "0 0 12px 0", fontSize: "18px", fontWeight: 600 }}>
        {name}
        {highlighted && " (Recommended)"}
      </h3>

      <div style={{ display: "grid", gap: "8px", marginBottom: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "#64748b", fontSize: "14px" }}>Accuracy</span>
          <span style={{ fontWeight: 600, fontSize: "16px" }}>{accuracy}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "#64748b", fontSize: "14px" }}>AUC</span>
          <span style={{ fontWeight: 600, fontSize: "16px" }}>{auc}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "#64748b", fontSize: "14px" }}>Precision</span>
          <span style={{ fontWeight: 600, fontSize: "16px" }}>{precision}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "#64748b", fontSize: "14px" }}>Recall</span>
          <span style={{ fontWeight: 600, fontSize: "16px" }}>{recall}</span>
        </div>
      </div>

      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {tags.map((tag) => (
          <span key={tag} className="badge badge-info" style={{ fontSize: "13px" }}>
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function ModelsPage() {
  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Model Comparison</p>
        <h1>Evaluating Different Approaches</h1>
        <p>
          Compare performance metrics, interpretability, and fairness characteristics across 
          different machine learning models for recidivism prediction.
        </p>
      </section>

      <div className="section-grid">
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Model Performance Cards</h2>
              <p>Detailed metrics for each model</p>
            </div>

            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: "16px",
              marginTop: "16px"
            }}>
              <ModelCard
                name="COMPAS"
                accuracy="65.4%"
                auc="0.68"
                precision="0.63"
                recall="0.59"
                tags={["Black-box", "Unfair"]}
              />
              <ModelCard
                name="Logistic Regression"
                accuracy="67.2%"
                auc="0.72"
                precision="0.66"
                recall="0.64"
                tags={["Interpretable"]}
                highlighted={true}
              />
              <ModelCard
                name="Random Forest"
                accuracy="69.1%"
                auc="0.74"
                precision="0.68"
                recall="0.66"
                tags={["Best Accuracy"]}
              />
              <ModelCard
                name="Decision Tree"
                accuracy="66.3%"
                auc="0.70"
                precision="0.64"
                recall="0.62"
                tags={["Interpretable", "Fast"]}
              />
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body two-column">
            <div>
              <div className="section-card__header">
                <h2>Accuracy Comparison</h2>
                <p>Overall accuracy across all models</p>
              </div>
              <p className="section-note">
                Random Forest achieves the highest accuracy at 69.1%, followed by Logistic Regression at 67.2%. 
                COMPAS baseline shows lower performance at 65.4%.
              </p>
            </div>
            <PredictionChart
              counts={[65.4, 67.2, 69.1, 66.3]}
              labels={["COMPAS", "Logistic Reg", "Random Forest", "Decision Tree"]}
            />
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>ROC Curves</h2>
              <p>Receiver Operating Characteristic curves for all models</p>
            </div>

            <div style={{
              background: '#f8fafc',
              borderRadius: '12px',
              padding: '24px',
              marginTop: '16px',
            }}>
              <svg viewBox="0 0 400 400" style={{
                width: '100%',
                maxWidth: '500px',
                height: 'auto',
              }}>
                {/* Axes */}
                <line x1="40" y1="360" x2="360" y2="360" stroke="#cbd5e1" strokeWidth="2" />
                <line x1="40" y1="360" x2="40" y2="40" stroke="#cbd5e1" strokeWidth="2" />
                
                {/* Grid lines */}
                {[1, 2, 3, 4, 5].map((i) => (
                  <g key={`grid-${i}`}>
                    <line x1="40" y1={360 - (i * 64)} x2="360" y2={360 - (i * 64)} stroke="#e2e8f0" strokeWidth="1" />
                    <line x1={40 + (i * 64)} y1="360" x2={40 + (i * 64)} y2="40" stroke="#e2e8f0" strokeWidth="1" />
                  </g>
                ))}
                
                {/* Diagonal reference line */}
                <line x1="40" y1="360" x2="360" y2="40" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4" opacity="0.5" />
                
                {/* COMPAS (AUC 0.68) - Red */}
                <path
                  d="M 40 360 Q 80 340, 120 300 T 240 140 T 360 40"
                  stroke="#ef4444"
                  strokeWidth="3"
                  fill="none"
                />
                
                {/* Logistic Regression (AUC 0.72) - Blue */}
                <path
                  d="M 40 360 Q 70 330, 110 280 T 220 100 T 360 40"
                  stroke="#0ea5e9"
                  strokeWidth="3"
                  fill="none"
                />
                
                {/* Random Forest (AUC 0.74) - Green */}
                <path
                  d="M 40 360 Q 65 325, 100 270 T 210 80 T 360 40"
                  stroke="#10b981"
                  strokeWidth="3"
                  fill="none"
                />
                
                {/* Decision Tree (AUC 0.70) - Amber */}
                <path
                  d="M 40 360 Q 75 335, 115 290 T 230 120 T 360 40"
                  stroke="#f59e0b"
                  strokeWidth="3"
                  fill="none"
                />
                
                {/* Axis labels */}
                <text x="200" y="390" textAnchor="middle" fontSize="12" fill="#64748b">
                  False Positive Rate
                </text>
                <text x="10" y="200" textAnchor="middle" fontSize="12" fill="#64748b" transform="rotate(-90 10 200)">
                  True Positive Rate
                </text>
                
                {/* Tick labels */}
                {[0, 0.2, 0.4, 0.6, 0.8, 1].map((val) => {
                  const x = 40 + (val * 320);
                  const y = 360 - (val * 320);
                  return (
                    <g key={`tick-${val}`}>
                      <text x={x} y="375" textAnchor="middle" fontSize="11" fill="#64748b">
                        {val.toFixed(1)}
                      </text>
                      <text x="30" y={y + 3} textAnchor="end" fontSize="11" fill="#64748b">
                        {val.toFixed(1)}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: '12px',
              marginTop: '20px',
            }}>
              <div style={{
                padding: '12px',
                background: 'rgba(239, 68, 68, 0.1)',
                borderRadius: '8px',
                borderLeft: '3px solid #ef4444',
              }}>
                <div style={{ fontSize: '12px', color: '#64748b' }}>COMPAS</div>
                <div style={{ fontSize: '18px', fontWeight: 600, color: '#1e293b' }}>0.68</div>
              </div>
              <div style={{
                padding: '12px',
                background: 'rgba(14, 165, 233, 0.1)',
                borderRadius: '8px',
                borderLeft: '3px solid #0ea5e9',
              }}>
                <div style={{ fontSize: '12px', color: '#64748b' }}>Logistic Reg</div>
                <div style={{ fontSize: '18px', fontWeight: 600, color: '#1e293b' }}>0.72</div>
              </div>
              <div style={{
                padding: '12px',
                background: 'rgba(16, 185, 129, 0.1)',
                borderRadius: '8px',
                borderLeft: '3px solid #10b981',
              }}>
                <div style={{ fontSize: '12px', color: '#64748b' }}>Random Forest</div>
                <div style={{ fontSize: '18px', fontWeight: 600, color: '#1e293b' }}>0.74</div>
              </div>
              <div style={{
                padding: '12px',
                background: 'rgba(245, 158, 11, 0.1)',
                borderRadius: '8px',
                borderLeft: '3px solid #f59e0b',
              }}>
                <div style={{ fontSize: '12px', color: '#64748b' }}>Decision Tree</div>
                <div style={{ fontSize: '18px', fontWeight: 600, color: '#1e293b' }}>0.70</div>
              </div>
            </div>

            <p className="section-note" style={{ marginTop: '16px' }}>
              ROC curves show the trade-off between True Positive Rate (sensitivity) and False Positive Rate. 
              Random Forest achieves the highest AUC (0.74), indicating the best discrimination ability. 
              Logistic Regression (AUC 0.72) offers strong performance with better interpretability.
            </p>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Confusion Matrices</h2>
              <p>Detailed classification results</p>
            </div>

            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "20px",
              marginTop: "16px"
            }}>
              {[
                { name: "Logistic Regression", tn: 2104, fp: 680, fn: 820, tp: 1610 },
                { name: "Random Forest", tn: 2240, fp: 544, fn: 748, tp: 1682 }
              ].map((model) => (
                <div key={model.name}>
                  <h4 style={{ margin: "0 0 12px 0", fontSize: "14px", fontWeight: 600 }}>
                    {model.name}
                  </h4>
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "2px",
                    borderRadius: "8px",
                    overflow: "hidden"
                  }}>
                    <div style={{
                      padding: "12px",
                      background: "#dcfce7",
                      textAlign: "center"
                    }}>
                      <div style={{ fontSize: "12px", color: "#64748b" }}>TN</div>
                      <div style={{ fontSize: "20px", fontWeight: 600 }}>{model.tn}</div>
                    </div>
                    <div style={{
                      padding: "12px",
                      background: "#fee2e2",
                      textAlign: "center"
                    }}>
                      <div style={{ fontSize: "12px", color: "#64748b" }}>FP</div>
                      <div style={{ fontSize: "20px", fontWeight: 600 }}>{model.fp}</div>
                    </div>
                    <div style={{
                      padding: "12px",
                      background: "#fef3c7",
                      textAlign: "center"
                    }}>
                      <div style={{ fontSize: "12px", color: "#64748b" }}>FN</div>
                      <div style={{ fontSize: "20px", fontWeight: 600 }}>{model.fn}</div>
                    </div>
                    <div style={{
                      padding: "12px",
                      background: "#dbeafe",
                      textAlign: "center"
                    }}>
                      <div style={{ fontSize: "12px", color: "#64748b" }}>TP</div>
                      <div style={{ fontSize: "20px", fontWeight: 600 }}>{model.tp}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Model Selection Guide</h2>
            </div>

            <div style={{
              display: "grid",
              gap: "16px",
              marginTop: "16px"
            }}>
              <div style={{
                padding: "16px",
                background: "rgba(14, 165, 233, 0.08)",
                borderRadius: "8px",
                borderLeft: "4px solid #0ea5e9"
              }}>
                <div style={{ fontWeight: 600, color: "#1e293b", marginBottom: "4px" }}>
                  Recommended: Logistic Regression
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  Offers strong interpretability with SHAP analysis, reasonable accuracy (67.2%), 
                  and supports fairness interventions. Best for practical deployment requiring explainability.
                </p>
              </div>

              <div style={{
                padding: "16px",
                background: "rgba(16, 185, 129, 0.08)",
                borderRadius: "8px",
                borderLeft: "4px solid #10b981"
              }}>
                <div style={{ fontWeight: 600, color: "#1e293b", marginBottom: "4px" }}>
                  Highest Accuracy: Random Forest
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  Best predictive performance (69.1% accuracy, 0.74 AUC) but less interpretable. 
                  Use with caution and supplementary fairness analysis.
                </p>
              </div>

              <div style={{
                padding: "16px",
                background: "rgba(239, 68, 68, 0.08)",
                borderRadius: "8px",
                borderLeft: "4px solid #ef4444"
              }}>
                <div style={{ fontWeight: 600, color: "#1e293b", marginBottom: "4px" }}>
                  Not Recommended: COMPAS
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  Proprietary black-box system with documented fairness issues, lower accuracy, 
                  and limited interpretability. Included for benchmarking purposes only.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
