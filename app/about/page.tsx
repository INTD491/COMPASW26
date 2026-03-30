"use client";

export default function AboutPage() {
  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">About</p>
        <h1>Understanding Fairness in Recidivism Prediction</h1>
        <p>
          This dashboard explores interpretability and fairness in machine learning models 
          for criminal risk assessment. We evaluate multiple approaches, their trade-offs, 
          and practical deployment considerations for high-stakes decision-making.
        </p>
      </section>

      <div className="section-grid">
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Project Overview</h2>
            </div>

            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
              gap: "16px",
              marginTop: "16px",
              marginBottom: "24px"
            }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "28px", fontWeight: 700, color: "#0ea5e9" }}>4</div>
                <div style={{ fontSize: "12px", color: "#64748b" }}>Research Questions</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "28px", fontWeight: 700, color: "#0ea5e9" }}>33K+</div>
                <div style={{ fontSize: "12px", color: "#64748b" }}>Data Records</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "28px", fontWeight: 700, color: "#0ea5e9" }}>5</div>
                <div style={{ fontSize: "12px", color: "#64748b" }}>Models Evaluated</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "28px", fontWeight: 700, color: "#0ea5e9" }}>2</div>
                <div style={{ fontSize: "12px", color: "#64748b" }}>States Analyzed</div>
              </div>
            </div>

            <div style={{
              padding: "16px",
              background: "#f8fafc",
              borderRadius: "8px",
              borderLeft: "4px solid #0ea5e9"
            }}>
              <p style={{ margin: 0, lineHeight: "1.6", color: "#1e293b", fontSize: "14px" }}>
                This project investigates how machine learning models trained on historical criminal justice 
                data can perpetuate and amplify systemic biases. We examine fairness metrics, debiasing techniques, 
                and the generalization of models across different jurisdictions.
              </p>
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Research Questions</h2>
            </div>

            <div style={{ display: "grid", gap: "12px", marginTop: "16px" }}>
              {[
                {
                  rq: "RQ1",
                  question: "Can we improve upon the COMPAS baseline with modern ML models?"
                },
                {
                  rq: "RQ2",
                  question: "What are the fairness trade-offs when optimizing for accuracy?"
                },
                {
                  rq: "RQ3",
                  question: "How do we provide interpretable explanations for high-stakes predictions?"
                },
                {
                  rq: "RQ4",
                  question: "Do models trained in one jurisdiction generalize to another?"
                }
              ].map((item) => (
                <div key={item.rq} style={{
                  padding: "12px",
                  background: "#f8fafc",
                  borderRadius: "8px",
                  borderLeft: "4px solid #0ea5e9"
                }}>
                  <span style={{
                    display: "inline-block",
                    padding: "2px 8px",
                    background: "#0ea5e9",
                    color: "white",
                    borderRadius: "4px",
                    fontSize: "12px",
                    fontWeight: 600,
                    marginRight: "8px"
                  }}>
                    {item.rq}
                  </span>
                  <span style={{ color: "#1e293b", fontSize: "14px" }}>
                    {item.question}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Data Sources</h2>
            </div>

            <div style={{ display: "grid", gap: "16px", marginTop: "16px" }}>
              <div style={{
                padding: "16px",
                border: "1px solid #e2e8f0",
                borderRadius: "8px"
              }}>
                <h3 style={{ margin: "0 0 8px 0", fontSize: "16px", fontWeight: 600 }}>
                  COMPAS Florida
                </h3>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b", lineHeight: "1.6" }}>
                  Proprietary Correctional Offender Management Profiling for Alternative Sanctions (COMPAS) 
                  dataset from Florida (2013-2014), containing 7,214 defendants and their 2-year recidivism outcomes.
                </p>
              </div>

              <div style={{
                padding: "16px",
                border: "1px solid #e2e8f0",
                borderRadius: "8px"
              }}>
                <h3 style={{ margin: "0 0 8px 0", fontSize: "16px", fontWeight: 600 }}>
                  NIJ Georgia
                </h3>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b", lineHeight: "1.6" }}>
                  National Institute of Justice (NIJ) dataset from Georgia, with 25,835 records used to evaluate 
                  model generalization and transferability across jurisdictions.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Technology Stack</h2>
            </div>

            <div style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "8px",
              marginTop: "16px"
            }}>
              {[
                "Python", "Scikit-learn", "XGBoost", "AIF360",
                "SHAP", "Pandas", "Next.js", "React",
                "Plotly", "TypeScript", "Vercel"
              ].map((tech) => (
                <span
                  key={tech}
                  className="badge badge-info"
                  style={{ fontSize: "13px" }}
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Key References</h2>
            </div>

            <div style={{
              display: "grid",
              gap: "12px",
              marginTop: "16px"
            }}>
              <div style={{ fontSize: "14px", lineHeight: "1.5" }}>
                <div style={{ fontWeight: 600, color: "#1e293b", marginBottom: "4px" }}>
                  Dressel, J., & Farid, H. (2018).
                </div>
                <p style={{ margin: 0, color: "#64748b" }}>
                  "The accuracy, fairness, and limits of predicting recidivism." 
                  Science Advances, 4(1), eaao5580.
                </p>
              </div>

              <div style={{ fontSize: "14px", lineHeight: "1.5" }}>
                <div style={{ fontWeight: 600, color: "#1e293b", marginBottom: "4px" }}>
                  Rudin, C., & Ustun, B. (2020).
                </div>
                <p style={{ margin: 0, color: "#64748b" }}>
                  "Optimized scoring systems: Toward trust in machine learning for 
                  criminal justice." Journal of Criminal Justice, 66, 101646.
                </p>
              </div>

              <div style={{ fontSize: "14px", lineHeight: "1.5" }}>
                <div style={{ fontWeight: 600, color: "#1e293b", marginBottom: "4px" }}>
                  Wang, Z., Xu, Z., & Glasser, J. (2022).
                </div>
                <p style={{ margin: 0, color: "#64748b" }}>
                  "Fair machine learning in criminal justice." 
                  Nature Machine Intelligence, 4(5), 408-415.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body" style={{
            background: "linear-gradient(135deg, #1e3a5f, #0f172a)",
            color: "white"
          }}>
            <div className="section-card__header">
              <h2 style={{ color: "white" }}>Project Information</h2>
            </div>

            <p style={{ margin: "0 0 12px 0", lineHeight: "1.6", opacity: "0.9" }}>
              This dashboard is part of INTD 491: Cloud Deployment, Winter 2026, 
              taught at the University of Alberta. It demonstrates best practices for 
              deploying ML models to the cloud with accountability and transparency.
            </p>

            <div style={{
              display: "grid",
              gap: "8px",
              fontSize: "14px"
            }}>
              <div>
                <strong>Course:</strong> INTD 491, Section B1
              </div>
              <div>
                <strong>Institution:</strong> University of Alberta
              </div>
              <div>
                <strong>Demo Date:</strong> March 30, 2026
              </div>
              <div>
                <strong>Deployment:</strong> Vercel Cloud Platform
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
