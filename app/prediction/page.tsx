"use client";

import { useState } from "react";
import { PredictionChart } from "@/components/PredictionChart";

export default function PredictionPage() {
  const [formData, setFormData] = useState({
    age: 34,
    sex: "Male",
    race: "African American",
    priorCrimes: 3,
    juvenileFelonies: 0,
    chargeDegree: "Felony",
    model: "Random Forest"
  });

  const [prediction, setPrediction] = useState<{
    riskScore: number;
    riskLevel: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: isNaN(Number(value)) ? value : Number(value)
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          age: formData.age,
          sex: formData.sex,
          race: formData.race,
          prior_crimes: formData.priorCrimes,
          juvenile_felonies: formData.juvenileFelonies,
          charge_degree: formData.chargeDegree,
          model: formData.model
        })
      });

      if (!response.ok) throw new Error("Prediction failed");

      const data = await response.json();
      setPrediction({
        riskScore: Math.round(data.risk_score * 100),
        riskLevel: data.risk_score > 0.6 ? "High" : data.risk_score > 0.3 ? "Medium" : "Low"
      });
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to get prediction");
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case "High": return "metric-red";
      case "Medium": return "metric-amber";
      case "Low": return "metric-green";
      default: return "metric-blue";
    }
  };

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Prediction Tool</p>
        <h1>Recidivism Risk Assessment</h1>
        <p>
          Enter demographic and criminal history information to generate a risk prediction 
          using multiple models. Compare predictions across algorithms and view feature importance.
        </p>
      </section>

      <div className="predict-grid">
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Input Information</h2>
              <p>Enter the individual's information</p>
            </div>

            <form className="input-form">
              <div className="form-group">
                <label className="form-label">Age</label>
                <input
                  className="form-input"
                  type="number"
                  name="age"
                  min="18"
                  max="80"
                  value={formData.age}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Sex</label>
                <select
                  className="form-select"
                  name="sex"
                  value={formData.sex}
                  onChange={handleInputChange}
                >
                  <option>Male</option>
                  <option>Female</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Race</label>
                <select
                  className="form-select"
                  name="race"
                  value={formData.race}
                  onChange={handleInputChange}
                >
                  <option>African American</option>
                  <option>Caucasian</option>
                  <option>Hispanic</option>
                  <option>Other</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Prior Crimes Count</label>
                <input
                  className="form-input"
                  type="number"
                  name="priorCrimes"
                  min="0"
                  max="40"
                  value={formData.priorCrimes}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Juvenile Felonies</label>
                <input
                  className="form-input"
                  type="number"
                  name="juvenileFelonies"
                  min="0"
                  max="10"
                  value={formData.juvenileFelonies}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Charge Degree</label>
                <select
                  className="form-select"
                  name="chargeDegree"
                  value={formData.chargeDegree}
                  onChange={handleInputChange}
                >
                  <option>Felony</option>
                  <option>Misdemeanor</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Model</label>
                <select
                  className="form-select"
                  name="model"
                  value={formData.model}
                  onChange={handleInputChange}
                >
                  <option>Logistic Regression</option>
                  <option>Random Forest</option>
                  <option>Decision Tree</option>
                  <option>XGBoost + Debiasing</option>
                </select>
              </div>

              <button
                className="button button--primary"
                type="button"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? "Predicting..." : "Get Prediction"}
              </button>
            </form>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Prediction Result</h2>
              <p>Risk assessment for this individual</p>
            </div>

            {prediction ? (
              <>
                <div className={`metric-card ${getRiskColor(prediction.riskLevel)}`}>
                  <div className="metric-label">Predicted Risk Level</div>
                  <div className="metric-value">{prediction.riskScore}%</div>
                  <div className="metric-subtext">{prediction.riskLevel} Risk</div>
                </div>

                <div style={{ marginTop: "24px" }}>
                  <div className="section-card__header">
                    <h3>Model Comparison</h3>
                    <p>Predictions across different models</p>
                  </div>
                  <PredictionChart
                    counts={[65.4, 67.2, 69.1, 66.3]}
                    labels={["COMPAS", "Logistic Reg", "Random Forest", "Decision Tree"]}
                  />
                </div>

                <div style={{ marginTop: "24px" }}>
                  <div className="section-card__header">
                    <h3>Interpretation</h3>
                  </div>
                  <p className="section-note">
                    This prediction is based on historical data from the COMPAS Florida dataset. 
                    The model considers prior criminal history, age, and other factors. 
                    Remember: models can reflect historical biases in the data and criminal justice system.
                  </p>
                </div>
              </>
            ) : (
              <div className="chart-placeholder">
                Enter information and click "Get Prediction" to see the risk assessment.
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
