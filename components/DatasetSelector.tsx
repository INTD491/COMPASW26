'use client';

import { Dataset } from '@/lib/dataLoader';

interface DatasetSelectorProps {
  selectedDataset: Dataset;
  onDatasetChange: (dataset: Dataset) => void;
}

interface DatasetInfo {
  value: Dataset;
  label: string;
  description: string;
  records: string;
}

const datasetInfo: DatasetInfo[] = [
  {
    value: 'florida',
    label: 'Florida (COMPAS)',
    description: 'ProPublica COMPAS dataset from Florida',
    records: '7,214 records',
  },
  {
    value: 'georgia',
    label: 'Georgia (NIJ)',
    description: 'National Institute of Justice recidivism data',
    records: '25,835 records',
  },
  {
    value: 'combined',
    label: 'Combined',
    description: 'Aggregate analysis across both jurisdictions',
    records: '33,049 records',
  },
];

export default function DatasetSelector({
  selectedDataset,
  onDatasetChange,
}: DatasetSelectorProps) {
  return (
    <div style={{ marginBottom: '32px' }}>
      <div style={{
        textAlign: 'center',
        marginBottom: '20px',
      }}>
        <h3 style={{
          fontSize: '16px',
          fontWeight: 600,
          color: '#1e293b',
          margin: '0 0 8px 0',
        }}>
          Select Jurisdiction Dataset
        </h3>
        <p style={{
          fontSize: '13px',
          color: '#64748b',
          margin: 0,
        }}>
          Choose a dataset to view jurisdiction-specific metrics and analysis
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '12px',
        maxWidth: '800px',
        margin: '0 auto',
      }}>
        {datasetInfo.map((dataset) => (
          <button
            key={dataset.value}
            onClick={() => onDatasetChange(dataset.value)}
            style={{
              padding: '16px',
              border: selectedDataset === dataset.value ? '2px solid #0ea5e9' : '2px solid #e2e8f0',
              borderRadius: '12px',
              background: selectedDataset === dataset.value ? 'rgba(14, 165, 233, 0.08)' : '#ffffff',
              cursor: 'pointer',
              transition: 'all 200ms ease',
              textAlign: 'left',
            }}
            onMouseEnter={(e) => {
              if (selectedDataset !== dataset.value) {
                (e.currentTarget as HTMLElement).style.borderColor = '#cbd5e1';
                (e.currentTarget as HTMLElement).style.background = '#f1f5f9';
              }
            }}
            onMouseLeave={(e) => {
              if (selectedDataset !== dataset.value) {
                (e.currentTarget as HTMLElement).style.borderColor = '#e2e8f0';
                (e.currentTarget as HTMLElement).style.background = '#ffffff';
              }
            }}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: '8px',
            }}>
              <div>
                <div style={{
                  fontSize: '15px',
                  fontWeight: 600,
                  color: '#1e293b',
                  marginBottom: '4px',
                }}>
                  {dataset.label}
                  {selectedDataset === dataset.value && (
                    <span style={{
                      display: 'inline-block',
                      marginLeft: '8px',
                      padding: '2px 8px',
                      background: '#0ea5e9',
                      color: 'white',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 600,
                    }}>
                      Active
                    </span>
                  )}
                </div>
                <p style={{
                  fontSize: '13px',
                  color: '#64748b',
                  margin: 0,
                }}>
                  {dataset.description}
                </p>
              </div>
            </div>
            <div style={{
              fontSize: '12px',
              color: '#0ea5e9',
              fontWeight: 500,
            }}>
              {dataset.records}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
