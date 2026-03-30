/**
 * Data loading utilities for COMPAS (Florida) and NIJ (Georgia) datasets
 * Handles CSV parsing and preprocessing for both jurisdictions
 */

export type Dataset = 'florida' | 'georgia' | 'combined';

export interface DemographicBreakdown {
  [key: string]: number;
}

export interface DatasetMetrics {
  totalRecords: number;
  recidivismRate: number;
  recidivismCount: number;
  meanAge: number;
  bestModelAUC: number;
  disparateImpact: number;
  race: DemographicBreakdown;
  raceRecidivism: {
    [key: string]: { count: number; recidivism: number; rate: number };
  };
  priorCrimesAvg: number;
  ageDistribution: DemographicBreakdown;
}

export interface COMPASRecord {
  id: string;
  age: number;
  race: string;
  sex: string;
  priors_count: number;
  two_year_recid: number;
  decile_score: number;
}

export interface GeorgiaRecord {
  ID: string;
  Age_at_Release: string;
  Race: string;
  Gender: string;
  Prior_Arrest_Episodes_Felony: number;
  Prior_Arrest_Episodes_Misd: number;
  Recidivism_Arrest_Year1: string;
  Recidivism_Arrest_Year2: string;
  Recidivism_Arrest_Year3: string;
  Supervision_Risk_Score_First: number;
}

/**
 * Parse COMPAS Florida data
 */
export async function loadFloridaData(): Promise<COMPASRecord[]> {
  try {
    const response = await fetch(
      '/data/compas-scores-two-years.csv'
    );
    const text = await response.text();
    const lines = text.split('\n');
    const headers = lines[0].split(',');

    const records: COMPASRecord[] = [];

    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;

      const values = lines[i].split(',');
      const record: any = {};

      headers.forEach((header, idx) => {
        record[header.trim()] = values[idx]?.trim();
      });

      // Only include records with valid target variable
      if (record.two_year_recid !== undefined && record.two_year_recid !== '') {
        records.push({
          id: record.id,
          age: parseInt(record.age) || 0,
          race: record.race || 'Unknown',
          sex: record.sex || 'Unknown',
          priors_count: parseInt(record.priors_count) || 0,
          two_year_recid: parseInt(record.two_year_recid) || 0,
          decile_score: parseInt(record.decile_score) || 0,
        });
      }
    }

    return records;
  } catch (error) {
    console.error('Error loading Florida data:', error);
    return [];
  }
}

/**
 * Parse Georgia NIJ data
 */
export async function loadGeorgiaData(): Promise<GeorgiaRecord[]> {
  try {
    const response = await fetch(
      '/data/nij-challenge2021_full_dataset.csv'
    );
    const text = await response.text();
    const lines = text.split('\n');
    const headers = lines[0].split(',');

    const records: GeorgiaRecord[] = [];

    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;

      const values = lines[i].split(',');
      const record: any = {};

      headers.forEach((header, idx) => {
        record[header.trim()] = values[idx]?.trim();
      });

      records.push({
        ID: record.ID,
        Age_at_Release: record.Age_at_Release || 'Unknown',
        Race: record.Race || 'Unknown',
        Gender: record.Gender || 'Unknown',
        Prior_Arrest_Episodes_Felony: parseInt(record.Prior_Arrest_Episodes_Felony) || 0,
        Prior_Arrest_Episodes_Misd: parseInt(record.Prior_Arrest_Episodes_Misd) || 0,
        Recidivism_Arrest_Year1: record.Recidivism_Arrest_Year1 || 'No',
        Recidivism_Arrest_Year2: record.Recidivism_Arrest_Year2 || 'No',
        Recidivism_Arrest_Year3: record.Recidivism_Arrest_Year3 || 'No',
        Supervision_Risk_Score_First: parseInt(record.Supervision_Risk_Score_First) || 0,
      });
    }

    return records;
  } catch (error) {
    console.error('Error loading Georgia data:', error);
    return [];
  }
}

/**
 * Calculate comprehensive metrics from COMPAS Florida data
 */
export function calculateFloridaMetrics(records: COMPASRecord[]): DatasetMetrics {
  if (records.length === 0) {
    return getEmptyMetrics();
  }

  const recidivismCount = records.reduce((sum, r) => sum + r.two_year_recid, 0);
  const recidivismRate = recidivismCount / records.length;

  // Race breakdown
  const raceMap: { [key: string]: number } = {};
  const raceRecidivismMap: { [key: string]: { count: number; recidivism: number } } = {};

  records.forEach((r) => {
    const race = r.race || 'Unknown';
    raceMap[race] = (raceMap[race] || 0) + 1;

    if (!raceRecidivismMap[race]) {
      raceRecidivismMap[race] = { count: 0, recidivism: 0 };
    }
    raceRecidivismMap[race].count += 1;
    raceRecidivismMap[race].recidivism += r.two_year_recid;
  });

  // Convert recidivism counts to rates
  const raceRecidivism: { [key: string]: { count: number; recidivism: number; rate: number } } = {};
  Object.entries(raceRecidivismMap).forEach(([race, data]) => {
    raceRecidivism[race] = {
      count: data.count,
      recidivism: data.recidivism,
      rate: data.recidivism / data.count,
    };
  });

  // Age distribution
  const ageDistribution: DemographicBreakdown = {
    '18-25': 0,
    '26-35': 0,
    '36-45': 0,
    '46-55': 0,
    '56+': 0,
  };

  records.forEach((r) => {
    if (r.age <= 25) ageDistribution['18-25']++;
    else if (r.age <= 35) ageDistribution['26-35']++;
    else if (r.age <= 45) ageDistribution['36-45']++;
    else if (r.age <= 55) ageDistribution['46-55']++;
    else ageDistribution['56+']++;
  });

  // Mean age
  const meanAge = records.reduce((sum, r) => sum + r.age, 0) / records.length;

  // Prior crimes average
  const priorCrimesAvg = records.reduce((sum, r) => sum + r.priors_count, 0) / records.length;

  // Disparate Impact (African-American recidivism vs White recidivism)
  const aaRecidivism = raceRecidivism['African-American']?.rate || 0;
  const whiteRecidivism = raceRecidivism['Caucasian']?.rate || raceRecidivism['White']?.rate || 0;
  const disparateImpact = whiteRecidivism > 0 ? aaRecidivism / whiteRecidivism : 0;

  return {
    totalRecords: records.length,
    recidivismRate,
    recidivismCount,
    meanAge,
    bestModelAUC: 0.74, // Placeholder - would need actual model predictions
    disparateImpact,
    race: raceMap,
    raceRecidivism,
    priorCrimesAvg,
    ageDistribution,
  };
}

/**
 * Calculate comprehensive metrics from Georgia NIJ data
 */
export function calculateGeorgiaMetrics(records: GeorgiaRecord[]): DatasetMetrics {
  if (records.length === 0) {
    return getEmptyMetrics();
  }

  // Consider recidivism within 3 years (any arrest in years 1-3)
  const recidivismCount = records.filter((r) => {
    return (
      r.Recidivism_Arrest_Year1?.toLowerCase() === 'yes' ||
      r.Recidivism_Arrest_Year2?.toLowerCase() === 'yes' ||
      r.Recidivism_Arrest_Year3?.toLowerCase() === 'yes'
    );
  }).length;

  const recidivismRate = recidivismCount / records.length;

  // Parse age ranges
  const ageMap: { [key: string]: number } = {
    '18-25': 0,
    '26-35': 0,
    '36-45': 0,
    '46-55': 0,
    '56+': 0,
  };

  records.forEach((r) => {
    const ageRange = r.Age_at_Release || 'Unknown';
    if (ageRange.includes('27') || ageRange === 'Less than 25') ageMap['18-25']++;
    else if (ageRange.includes('33') || ageRange.includes('37')) ageMap['26-35']++;
    else if (ageRange.includes('38') || ageRange.includes('42')) ageMap['36-45']++;
    else if (ageRange.includes('43') || ageRange.includes('47')) ageMap['46-55']++;
    else ageMap['56+']++;
  });

  // Race breakdown
  const raceMap: { [key: string]: number } = {};
  const raceRecidivismMap: { [key: string]: { count: number; recidivism: number } } = {};

  records.forEach((r) => {
    const race = r.Race || 'Unknown';
    raceMap[race] = (raceMap[race] || 0) + 1;

    const isRecidivist =
      r.Recidivism_Arrest_Year1?.toLowerCase() === 'yes' ||
      r.Recidivism_Arrest_Year2?.toLowerCase() === 'yes' ||
      r.Recidivism_Arrest_Year3?.toLowerCase() === 'yes'
        ? 1
        : 0;

    if (!raceRecidivismMap[race]) {
      raceRecidivismMap[race] = { count: 0, recidivism: 0 };
    }
    raceRecidivismMap[race].count += 1;
    raceRecidivismMap[race].recidivism += isRecidivist;
  });

  const raceRecidivism: { [key: string]: { count: number; recidivism: number; rate: number } } = {};
  Object.entries(raceRecidivismMap).forEach(([race, data]) => {
    raceRecidivism[race] = {
      count: data.count,
      recidivism: data.recidivism,
      rate: data.recidivism / data.count,
    };
  });

  // Mean age (estimate from ranges)
  const meanAge = 42; // Placeholder - would need full age values

  // Prior crimes average
  const priorCrimesAvg =
    records.reduce((sum, r) => sum + r.Prior_Arrest_Episodes_Felony + r.Prior_Arrest_Episodes_Misd, 0) /
    records.length;

  // Disparate Impact
  const blackRecidivism = raceRecidivism['BLACK']?.rate || 0;
  const whiteRecidivism = raceRecidivism['WHITE']?.rate || 0;
  const disparateImpact = whiteRecidivism > 0 ? blackRecidivism / whiteRecidivism : 0;

  return {
    totalRecords: records.length,
    recidivismRate,
    recidivismCount,
    meanAge,
    bestModelAUC: 0.69, // Placeholder - would need actual model predictions
    disparateImpact,
    race: raceMap,
    raceRecidivism,
    priorCrimesAvg,
    ageDistribution: ageMap,
  };
}

/**
 * Return empty metrics structure
 */
function getEmptyMetrics(): DatasetMetrics {
  return {
    totalRecords: 0,
    recidivismRate: 0,
    recidivismCount: 0,
    meanAge: 0,
    bestModelAUC: 0,
    disparateImpact: 0,
    race: {},
    raceRecidivism: {},
    priorCrimesAvg: 0,
    ageDistribution: {},
  };
}

/**
 * Get metrics for selected dataset
 */
export async function getDatasetMetrics(dataset: Dataset): Promise<DatasetMetrics> {
  if (dataset === 'florida') {
    const records = await loadFloridaData();
    return calculateFloridaMetrics(records);
  } else if (dataset === 'georgia') {
    const records = await loadGeorgiaData();
    return calculateGeorgiaMetrics(records);
  } else {
    // Combined - would need to aggregate both datasets
    const floridaRecords = await loadFloridaData();
    const georgiaRecords = await loadGeorgiaData();
    const floridaMetrics = calculateFloridaMetrics(floridaRecords);
    const georgiaMetrics = calculateGeorgiaMetrics(georgiaRecords);

    // Simple combination (would need more sophisticated aggregation)
    return {
      totalRecords: floridaMetrics.totalRecords + georgiaMetrics.totalRecords,
      recidivismRate:
        (floridaMetrics.recidivismCount + georgiaMetrics.recidivismCount) /
        (floridaMetrics.totalRecords + georgiaMetrics.totalRecords),
      recidivismCount: floridaMetrics.recidivismCount + georgiaMetrics.recidivismCount,
      meanAge: (floridaMetrics.meanAge + georgiaMetrics.meanAge) / 2,
      bestModelAUC: (floridaMetrics.bestModelAUC + georgiaMetrics.bestModelAUC) / 2,
      disparateImpact: (floridaMetrics.disparateImpact + georgiaMetrics.disparateImpact) / 2,
      race: { ...floridaMetrics.race, ...georgiaMetrics.race },
      raceRecidivism: { ...floridaMetrics.raceRecidivism, ...georgiaMetrics.raceRecidivism },
      priorCrimesAvg: (floridaMetrics.priorCrimesAvg + georgiaMetrics.priorCrimesAvg) / 2,
      ageDistribution: floridaMetrics.ageDistribution,
    };
  }
}
