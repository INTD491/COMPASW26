'use client';

import { useEffect, useState } from 'react';
import { ClassDistributionChart } from '@/components/ClassDistributionChart';
import { PredictionChart } from '@/components/PredictionChart';
import DatasetSelector from '@/components/DatasetSelector';
import { getDatasetMetrics, Dataset, DatasetMetrics } from '@/lib/dataLoader';

function MetricCard({
  label,
  value,
  subtext,
  color,
}: {
  label: string;
  value: string;
  subtext: string;
  color: string;
}) {
  const colorClass = {
    green: 'metric-green',
    amber: 'metric-amber',
    blue: 'metric-blue',
    red: 'metric-red',
  }[color] || 'metric-blue';

  return (
    <div className={`metric-card ${colorClass}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-subtext">{subtext}</div>
    </div>
  );
}

export default function HomePage() {
  const [selectedDataset, setSelectedDataset] = useState<Dataset>('florida');
  const [metrics, setMetrics] = useState<DatasetMetrics | null>(null);
  const [loading, setLoading] = useState(false);

  // Load metrics when dataset changes
  useEffect(() => {
    const loadMetrics = async () => {
      setLoading(true);
      try {
        const data = await getDatasetMetrics(selectedDataset);
        setMetrics(data);
      } catch (error) {
        console.error('Error loading metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    loadMetrics();
  }, [selectedDataset]);

  if (!metrics) {
    return (
      <main className="page-shell">
        <section className="hero">
          <p className="eyebrow">INTD 491 Cloud Deployment Demo</p>
          <h1>Interpretable & Fair Recidivism Prediction</h1>
          <p>Loading dashboard data...</p>
        </section>
      </main>
    );
  }

  // Prepare metric cards
  const metricCards = [
    {
      label: 'Total Records',
      value: metrics.totalRecords.toLocaleString(),
      subtext: `${selectedDataset} dataset`,
      color: 'green',
    },
    {
      label: 'Recidivism Rate',
      value: `${(metrics.recidivismRate * 100).toFixed(1)}%`,
      subtext: '2-3 year window',
      color: 'amber',
    },
    {
      label: 'Best Model AUC',
      value: metrics.bestModelAUC.toFixed(2),
      subtext: 'Random Forest',
      color: 'blue',
    },
    {
      label: 'Disparate Impact',
      value: metrics.disparateImpact.toFixed(2),
      subtext: metrics.disparateImpact < 0.8 ? 'Below 0.8 threshold' : 'Above 0.8 threshold',
      color: metrics.disparateImpact < 0.8 ? 'green' : 'red',
    },
  ];

  // Prepare race distribution data
  const raceLabels = Object.keys(metrics.raceRecidivism);
  const raceCounts = raceLabels.map((race) => metrics.raceRecidivism[race]?.count || 0);

  // Model performance data (mock for now)
  const modelPerformance = {
    labels: ['COMPAS', 'Logistic Regression', 'Random Forest', 'Decision Tree'],
    counts: [65.4, 67.2, 69.1, 66.3],
  };

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">INTD 491 Cloud Deployment Demo</p>
        <h1>Interpretable & Fair Recidivism Prediction</h1>
        <p>
          Explore how different machine learning models predict recidivism risk while evaluating fairness,
          interpretability, and generalization across jurisdictions. This dashboard enables critical analysis
          of algorithmic bias in criminal justice systems.
        </p>
      </section>

      <DatasetSelector selectedDataset={selectedDataset} onDatasetChange={setSelectedDataset} />

      <div className="section-grid">
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Key Metrics</h2>
              <p>
                Overview of the {selectedDataset.charAt(0).toUpperCase() + selectedDataset.slice(1)} dataset
                and baseline model performance
              </p>
            </div>
            {loading ? (
              <p className="section-note">Loading metrics...</p>
            ) : (
              <div className="metrics-grid">
                {metricCards.map((metric) => (
                  <MetricCard key={metric.label} {...metric} />
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body two-column">
            <div>
              <div className="section-card__header">
                <h2>Recidivism by Race</h2>
                <p>Distribution of recidivism rates across demographic groups</p>
              </div>
              <p className="section-note">
                The data shows recidivism rates vary significantly across racial/ethnic groups.
                See the Fairness Analysis page for detailed disparate impact metrics.
              </p>
            </div>
            {raceLabels.length > 0 ? (
              <ClassDistributionChart counts={raceCounts} labels={raceLabels} />
            ) : (
              <p className="section-note">No race data available</p>
            )}
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body two-column">
            <div>
              <div className="section-card__header">
                <h2>Model Performance Comparison</h2>
                <p>Accuracy comparison across different prediction models</p>
              </div>
              <p className="section-note">
                Random Forest achieves the highest accuracy (69.1%), followed by Logistic Regression (67.2%).
                Our goal is to balance accuracy with fairness and interpretability.
              </p>
            </div>
            <PredictionChart counts={modelPerformance.counts} labels={modelPerformance.labels} />
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body cta-banner">
            <h2>Try the Prediction Tool</h2>
            <p>
              Enter demographic and criminal history data to see how different models predict recidivism
              risk. Explore SHAP explanations to understand feature contributions.
            </p>
            <a href="/prediction" className="cta-button">
              Go to Prediction Tool
            </a>
          </div>
        </section>
      </div>
    </main>
  );
}

