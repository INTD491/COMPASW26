// Exploratory Data Analysis statistics — all computation runs client-side
// from CSV files served under /public/data/

export interface NumericStat {
  variable: string;
  label: string;
  count: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
  skewness: number;   // Pearson's skewness: (mean - median) / std
  outliers: number;   // count beyond Q1-1.5*IQR or Q3+1.5*IQR
}

export interface CategoryCount {
  label: string;
  count: number;
  pct: number;
}

export interface CategoricalStat {
  variable: string;
  label: string;
  categories: CategoryCount[];
  mode: string;
  total: number;
}

export interface MissingDataStat {
  variable: string;
  missing: number;
  pct: number;
}

export interface CorrelationStat {
  labelA: string;
  labelB: string;
  r: number;
  strength: "strong" | "moderate" | "weak" | "negligible";
  direction: "positive" | "negative" | "none";
}

export interface EDAResult {
  datasetName: string;
  totalRecords: number;
  totalColumns: number;
  recidivismRate: number;
  numericStats: NumericStat[];
  categoricalStats: CategoricalStat[];
  missingData: MissingDataStat[];
  correlations: CorrelationStat[];
}

// --- CSV Parsing ---

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split("\n");
  if (lines.length === 0) return [];

  const parseRow = (line: string): string[] => {
    const result: string[] = [];
    let inQuote = false;
    let current = "";
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        inQuote = !inQuote;
      } else if (ch === "," && !inQuote) {
        result.push(current.trim());
        current = "";
      } else {
        current += ch;
      }
    }
    result.push(current.trim());
    return result;
  };

  const headers = parseRow(lines[0]);
  return lines
    .slice(1)
    .filter((l) => l.trim())
    .map((line) => {
      const values = parseRow(line);
      const record: Record<string, string> = {};
      headers.forEach((h, i) => {
        record[h] = values[i] ?? "";
      });
      return record;
    });
}

// --- Statistical Helpers ---

function numericVals(records: Record<string, string>[], col: string): number[] {
  return records
    .map((r) => parseFloat(r[col]))
    .filter((v) => !isNaN(v));
}

function computeNumericStat(
  records: Record<string, string>[],
  col: string,
  label: string
): NumericStat {
  const vals = numericVals(records, col).sort((a, b) => a - b);
  const n = vals.length;
  if (n === 0)
    return { variable: col, label, count: 0, mean: 0, median: 0, std: 0, min: 0, max: 0, q1: 0, q3: 0, skewness: 0, outliers: 0 };

  const mean = vals.reduce((s, v) => s + v, 0) / n;
  const median = n % 2 === 0 ? (vals[n / 2 - 1] + vals[n / 2]) / 2 : vals[Math.floor(n / 2)];
  const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  const q1 = vals[Math.floor(n * 0.25)];
  const q3 = vals[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const outliers = vals.filter((v) => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr).length;
  const skewness = std > 0 ? (mean - median) / std : 0;

  return {
    variable: col,
    label,
    count: n,
    mean: +mean.toFixed(2),
    median: +median.toFixed(2),
    std: +std.toFixed(2),
    min: vals[0],
    max: vals[n - 1],
    q1: +q1.toFixed(2),
    q3: +q3.toFixed(2),
    skewness: +skewness.toFixed(3),
    outliers,
  };
}

function computeCategoricalStat(
  records: Record<string, string>[],
  col: string,
  label: string
): CategoricalStat {
  const counts: Record<string, number> = {};
  let total = 0;
  for (const r of records) {
    const v = r[col]?.trim();
    if (v !== undefined && v !== "") {
      counts[v] = (counts[v] || 0) + 1;
      total++;
    }
  }
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const categories = sorted.slice(0, 10).map(([lbl, count]) => ({
    label: lbl,
    count,
    pct: +((count / total) * 100).toFixed(1),
  }));
  return { variable: col, label, categories, mode: sorted[0]?.[0] ?? "", total };
}

function computeMissing(
  records: Record<string, string>[],
  col: string
): MissingDataStat {
  const total = records.length;
  const missing = records.filter((r) => {
    const v = r[col];
    return v === undefined || v === "" || v.toLowerCase() === "null" || v.toUpperCase() === "NA";
  }).length;
  return { variable: col, missing, pct: +((missing / total) * 100).toFixed(1) };
}

function pearsonR(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n === 0) return 0;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  const num = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
  const denX = Math.sqrt(xs.reduce((s, x) => s + (x - mx) ** 2, 0));
  const denY = Math.sqrt(ys.reduce((s, y) => s + (y - my) ** 2, 0));
  return denX && denY ? +(num / (denX * denY)).toFixed(3) : 0;
}

function correlationStat(
  records: Record<string, string>[],
  colA: string,
  labelA: string,
  colB: string,
  labelB: string,
  bNumeric = true
): CorrelationStat {
  const pairs = records
    .map((r) => {
      const a = parseFloat(r[colA]);
      const bRaw = r[colB]?.trim().toLowerCase();
      const b = bNumeric
        ? parseFloat(r[colB])
        : bRaw === "true" || bRaw === "1" || bRaw === "yes"
        ? 1
        : bRaw === "false" || bRaw === "0" || bRaw === "no"
        ? 0
        : NaN;
      return [a, b] as [number, number];
    })
    .filter(([a, b]) => !isNaN(a) && !isNaN(b));

  const r = pearsonR(
    pairs.map(([a]) => a),
    pairs.map(([, b]) => b)
  );
  const abs = Math.abs(r);
  const strength: CorrelationStat["strength"] =
    abs >= 0.5 ? "strong" : abs >= 0.3 ? "moderate" : abs >= 0.1 ? "weak" : "negligible";
  const direction: CorrelationStat["direction"] =
    r > 0.01 ? "positive" : r < -0.01 ? "negative" : "none";

  return { labelA, labelB, r, strength, direction };
}

// --- Florida COMPAS EDA ---

export async function computeFloridaEDA(): Promise<EDAResult> {
  const res = await fetch("/data/compas-scores-two-years.csv");
  const text = await res.text();
  const records = parseCSV(text);

  const numericCols: [string, string][] = [
    ["age", "Age"],
    ["priors_count", "Prior Crimes Count"],
    ["decile_score", "COMPAS Decile Score"],
    ["juv_fel_count", "Juvenile Felonies"],
    ["juv_misd_count", "Juvenile Misdemeanors"],
    ["juv_other_count", "Juvenile Other Charges"],
  ];

  const categoricalCols: [string, string][] = [
    ["race", "Race"],
    ["sex", "Sex"],
    ["age_cat", "Age Category"],
    ["c_charge_degree", "Charge Degree"],
    ["score_text", "COMPAS Risk Level"],
  ];

  const allCols = [...numericCols.map(([c]) => c), ...categoricalCols.map(([c]) => c), "two_year_recid"];

  const recidVals = records.map((r) => parseInt(r["two_year_recid"])).filter((v) => !isNaN(v));
  const recidivismRate = recidVals.filter((v) => v === 1).length / recidVals.length;

  const numericStats = numericCols.map(([col, label]) => computeNumericStat(records, col, label));
  const categoricalStats = categoricalCols.map(([col, label]) => computeCategoricalStat(records, col, label));
  const missingData = allCols
    .map((col) => computeMissing(records, col))
    .filter((m) => m.missing > 0);

  const correlations: CorrelationStat[] = [
    correlationStat(records, "priors_count", "Prior Crimes Count", "two_year_recid", "2-Year Recidivism"),
    correlationStat(records, "age", "Age", "two_year_recid", "2-Year Recidivism"),
    correlationStat(records, "decile_score", "COMPAS Decile Score", "two_year_recid", "2-Year Recidivism"),
    correlationStat(records, "decile_score", "COMPAS Decile Score", "priors_count", "Prior Crimes Count"),
    correlationStat(records, "juv_fel_count", "Juvenile Felonies", "two_year_recid", "2-Year Recidivism"),
  ];

  return {
    datasetName: "Florida COMPAS",
    totalRecords: records.length,
    totalColumns: Object.keys(records[0] ?? {}).length,
    recidivismRate: +recidivismRate.toFixed(4),
    numericStats,
    categoricalStats,
    missingData,
    correlations,
  };
}

// --- Georgia NIJ EDA ---

export async function computeGeorgiaEDA(): Promise<EDAResult> {
  const res = await fetch("/data/nij-challenge2021_full_dataset.csv");
  const text = await res.text();
  const records = parseCSV(text);

  const numericCols: [string, string][] = [
    ["Supervision_Risk_Score_First", "Supervision Risk Score"],
    ["Prior_Arrest_Episodes_Felony", "Prior Felony Arrests"],
    ["Prior_Arrest_Episodes_Misd", "Prior Misdemeanor Arrests"],
    ["Prior_Arrest_Episodes_Violent", "Prior Violent Arrests"],
    ["Prior_Arrest_Episodes_Property", "Prior Property Arrests"],
    ["Prior_Arrest_Episodes_Drug", "Prior Drug Arrests"],
    ["Prior_Conviction_Episodes_Felony", "Prior Felony Convictions"],
    ["Prior_Conviction_Episodes_Misd", "Prior Misdemeanor Convictions"],
    ["Percent_Days_Employed", "% Days Employed"],
    ["Delinquency_Reports", "Delinquency Reports"],
    ["Program_Attendances", "Program Attendances"],
  ];

  const categoricalCols: [string, string][] = [
    ["Age_at_Release", "Age at Release"],
    ["Race", "Race"],
    ["Gender", "Gender"],
    ["Education_Level", "Education Level"],
    ["Gang_Affiliated", "Gang Affiliated"],
    ["Prison_Years", "Years in Prison"],
  ];

  const allCols = [...numericCols.map(([c]) => c), ...categoricalCols.map(([c]) => c), "Recidivism_Within_3years"];

  const recidVals = records.map((r) => r["Recidivism_Within_3years"]?.trim().toLowerCase());
  const recidCount = recidVals.filter((v) => v === "true" || v === "1" || v === "yes").length;
  const recidivismRate = recidCount / records.length;

  const numericStats = numericCols.map(([col, label]) => computeNumericStat(records, col, label));
  const categoricalStats = categoricalCols.map(([col, label]) => computeCategoricalStat(records, col, label));
  const missingData = allCols
    .map((col) => computeMissing(records, col))
    .filter((m) => m.missing > 0);

  const correlations: CorrelationStat[] = [
    correlationStat(records, "Supervision_Risk_Score_First", "Supervision Risk Score", "Recidivism_Within_3years", "3-Year Recidivism", false),
    correlationStat(records, "Prior_Arrest_Episodes_Felony", "Prior Felony Arrests", "Recidivism_Within_3years", "3-Year Recidivism", false),
    correlationStat(records, "Percent_Days_Employed", "% Days Employed", "Recidivism_Within_3years", "3-Year Recidivism", false),
    correlationStat(records, "Supervision_Risk_Score_First", "Supervision Risk Score", "Delinquency_Reports", "Delinquency Reports"),
    correlationStat(records, "Prior_Arrest_Episodes_Felony", "Prior Felony Arrests", "Prior_Conviction_Episodes_Felony", "Prior Felony Convictions"),
  ];

  return {
    datasetName: "Georgia NIJ",
    totalRecords: records.length,
    totalColumns: Object.keys(records[0] ?? {}).length,
    recidivismRate: +recidivismRate.toFixed(4),
    numericStats,
    categoricalStats,
    missingData,
    correlations,
  };
}
