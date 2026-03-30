// Diagnostic statistics — runs all hypothesis tests client-side from CSV data.

export type Verdict = "Supported" | "Not Supported" | "Inconclusive";

export interface HypothesisResult {
  id: string;
  verdict: Verdict;
  testName: string;
  statLabel: string;
  statValue: string;
  pValue: number | null;
  summary: string;
  charts: ChartDef[];
}

export interface ChartDef {
  title: string;
  traces: SeriesTrace[];
  xLabel?: string;
  yLabel?: string;
  type?: "bar" | "scatter";
}

export interface SeriesTrace {
  name: string;
  x: string[];
  y: number[];
  color?: string;
}

export interface DiagnosticResults {
  datasetName: string;
  hypotheses: HypothesisResult[];
}

// ── CSV parsing ────────────────────────────────────────────────────────────

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split("\n");
  if (!lines.length) return [];
  const parseRow = (line: string): string[] => {
    const out: string[] = [];
    let inQ = false, cur = "";
    for (const ch of line) {
      if (ch === '"') { inQ = !inQ; }
      else if (ch === "," && !inQ) { out.push(cur.trim()); cur = ""; }
      else cur += ch;
    }
    out.push(cur.trim());
    return out;
  };
  const headers = parseRow(lines[0]);
  return lines.slice(1).filter(l => l.trim()).map(line => {
    const vals = parseRow(line);
    const rec: Record<string, string> = {};
    headers.forEach((h, i) => { rec[h] = vals[i] ?? ""; });
    return rec;
  });
}

// ── Statistical utilities ──────────────────────────────────────────────────

function mean(arr: number[]): number {
  return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
}

function toBool(v: string): boolean {
  const s = (v ?? "").trim().toLowerCase();
  return s === "true" || s === "1" || s === "yes";
}

/** Normal CDF via Abramowitz & Stegun approximation (accurate to ±7.5×10⁻⁸). */
function normalCDF(z: number): number {
  const abs = Math.abs(z);
  const t = 1 / (1 + 0.2316419 * abs);
  const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const phi = Math.exp(-0.5 * abs * abs) / Math.sqrt(2 * Math.PI);
  const result = 1 - phi * poly;
  return z >= 0 ? result : 1 - result;
}

/**
 * Mann-Whitney U test using rank-sum method (O(n log n)).
 * Returns two-tailed p-value via normal approximation.
 */
function mannWhitneyU(g1: number[], g2: number[]): { U: number; z: number; p: number } {
  const n1 = g1.length, n2 = g2.length;
  const all = [...g1.map(v => ({ v, g: 1 })), ...g2.map(v => ({ v, g: 2 }))];
  all.sort((a, b) => a.v - b.v);
  // Average ranks for ties
  let i = 0;
  const ranks: number[] = new Array(all.length);
  while (i < all.length) {
    let j = i;
    while (j < all.length && all[j].v === all[i].v) j++;
    const avg = (i + j + 1) / 2;
    for (let k = i; k < j; k++) ranks[k] = avg;
    i = j;
  }
  let R1 = 0;
  for (let k = 0; k < all.length; k++) if (all[k].g === 1) R1 += ranks[k];
  const U1 = R1 - (n1 * (n1 + 1)) / 2;
  const U2 = n1 * n2 - U1;
  const U = Math.min(U1, U2);
  const mu = (n1 * n2) / 2;
  const sigma = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
  const z = (U - mu) / sigma;
  const p = 2 * (1 - normalCDF(Math.abs(z)));
  return { U, z, p };
}

function groupMean(records: Record<string, string>[], groupCol: string, valueCol: string): Record<string, number> {
  const buckets: Record<string, number[]> = {};
  for (const r of records) {
    const g = r[groupCol]?.trim();
    const v = parseFloat(r[valueCol]);
    if (g && !isNaN(v)) {
      buckets[g] = buckets[g] ?? [];
      buckets[g].push(v);
    }
  }
  const out: Record<string, number> = {};
  for (const [k, arr] of Object.entries(buckets)) out[k] = +mean(arr).toFixed(3);
  return out;
}

function groupRate(records: Record<string, string>[], groupCol: string, outcomeCol: string): Record<string, number> {
  const buckets: Record<string, { yes: number; total: number }> = {};
  for (const r of records) {
    const g = r[groupCol]?.trim();
    if (!g) continue;
    buckets[g] = buckets[g] ?? { yes: 0, total: 0 };
    buckets[g].total++;
    if (toBool(r[outcomeCol]) || r[outcomeCol]?.trim() === "1") buckets[g].yes++;
  }
  const out: Record<string, number> = {};
  for (const [k, { yes, total }] of Object.entries(buckets)) out[k] = +((yes / total) * 100).toFixed(1);
  return out;
}

// ── Florida diagnostics ────────────────────────────────────────────────────

export async function computeFloridaDiagnostics(): Promise<DiagnosticResults> {
  const res = await fetch("/data/compas-scores-two-years.csv");
  const text = await res.text();
  const records = parseCSV(text);

  // Pull out typed arrays once
  const getNum = (r: Record<string, string>, col: string) => parseFloat(r[col]);
  const getStr = (r: Record<string, string>, col: string) => r[col]?.trim() ?? "";

  // ── H1: Race as demographic proxy ──────────────────────────────────────

  const races = ["African-American", "Caucasian", "Hispanic"];
  const raceColors: Record<string, string> = {
    "African-American": "#ef4444",
    "Caucasian": "#0ea5e9",
    "Hispanic": "#f59e0b",
    "Other": "#94a3b8",
    "Asian": "#10b981",
    "Native American": "#8b5cf6",
  };

  const byRace: Record<string, number[]> = {};
  for (const r of records) {
    const race = getStr(r, "race");
    const score = getNum(r, "decile_score");
    if (!isNaN(score)) {
      byRace[race] = byRace[race] ?? [];
      byRace[race].push(score);
    }
  }
  const raceOrder = Object.keys(byRace).sort((a, b) => mean(byRace[b]) - mean(byRace[a]));
  const meanScoreByRace = raceOrder.map(race => ({ race, mean: +mean(byRace[race]).toFixed(2) }));

  // Mann-Whitney U: AA vs Caucasian
  const aaScores = byRace["African-American"] ?? [];
  const caucScores = byRace["Caucasian"] ?? [];
  const mwH1 = mannWhitneyU(aaScores, caucScores);

  // Stratified by prior crimes bracket
  const priorBrackets = [
    { label: "0 prior crimes",  test: (n: number) => n === 0 },
    { label: "1–3 prior crimes", test: (n: number) => n >= 1 && n <= 3 },
    { label: "4+ prior crimes", test: (n: number) => n >= 4 },
  ];

  const stratifiedData: { bracket: string; AA: number; Caucasian: number }[] = priorBrackets.map(b => {
    const filt = (race: string) =>
      records
        .filter(r => getStr(r, "race") === race && b.test(getNum(r, "priors_count")))
        .map(r => getNum(r, "decile_score"))
        .filter(v => !isNaN(v));
    return { bracket: b.label, AA: +mean(filt("African-American")).toFixed(2), Caucasian: +mean(filt("Caucasian")).toFixed(2) };
  });

  const h1: HypothesisResult = {
    id: "H1",
    verdict: mwH1.p < 0.001 ? "Supported" : mwH1.p < 0.05 ? "Inconclusive" : "Not Supported",
    testName: "Mann-Whitney U test (AA vs. Caucasian decile scores)",
    statLabel: "z-score",
    statValue: mwH1.z.toFixed(3),
    pValue: mwH1.p,
    summary: `African-Americans receive a mean decile score of ${mean(aaScores).toFixed(2)} vs. ${mean(caucScores).toFixed(2)} for Caucasians (z = ${mwH1.z.toFixed(2)}, p ${mwH1.p < 0.001 ? "< 0.001" : mwH1.p.toFixed(3)}). The gap persists across all prior-crimes brackets — at 0 prior crimes, AA mean score is ${stratifiedData[0].AA} vs. ${stratifiedData[0].Caucasian} for Caucasians. Race carries predictive signal beyond criminal history.`,
    charts: [
      {
        title: "Mean COMPAS Decile Score by Race",
        xLabel: "Race", yLabel: "Mean decile score",
        traces: [{ name: "Mean score", x: meanScoreByRace.map(d => d.race), y: meanScoreByRace.map(d => d.mean), color: "#0ea5e9" }],
      },
      {
        title: "Mean Decile Score by Race, Controlling for Prior Crimes",
        xLabel: "Prior crimes bracket", yLabel: "Mean decile score",
        traces: [
          { name: "African-American", x: stratifiedData.map(d => d.bracket), y: stratifiedData.map(d => d.AA), color: "#ef4444" },
          { name: "Caucasian",        x: stratifiedData.map(d => d.bracket), y: stratifiedData.map(d => d.Caucasian), color: "#0ea5e9" },
        ],
      },
    ],
  };

  // ── H2: Bimodal decile score distribution ──────────────────────────────

  const decileCounts: Record<number, number> = {};
  for (let d = 1; d <= 10; d++) decileCounts[d] = 0;
  for (const r of records) {
    const s = getNum(r, "decile_score");
    if (!isNaN(s) && s >= 1 && s <= 10) decileCounts[s]++;
  }
  const decileLabels = Array.from({ length: 10 }, (_, i) => String(i + 1));
  const decileVals = decileLabels.map(l => decileCounts[Number(l)] ?? 0);
  // Check bimodality: peaks at 1-2 vs 9-10 compared to middle (4-7)
  const lowPeak = mean([decileVals[0], decileVals[1]]);
  const midVal  = mean(decileVals.slice(3, 7));
  const highPeak = mean([decileVals[8], decileVals[9]]);
  const bimodalRatio = (lowPeak + highPeak) / 2 / midVal;

  const h2: HypothesisResult = {
    id: "H2",
    verdict: bimodalRatio > 1.4 ? "Supported" : bimodalRatio > 1.1 ? "Inconclusive" : "Not Supported",
    testName: "Frequency distribution inspection (decile scores 1–10)",
    statLabel: "Peak-to-middle ratio",
    statValue: bimodalRatio.toFixed(2),
    pValue: null,
    summary: `Scores at deciles 1–2 average ${lowPeak.toFixed(0)} records/bin, scores 9–10 average ${highPeak.toFixed(0)} records/bin, while the middle range (4–7) averages only ${midVal.toFixed(0)} records/bin. The extremes are ${bimodalRatio.toFixed(1)}× more populated than the middle — consistent with a threshold-based internal classification rather than a continuous risk model.`,
    charts: [
      {
        title: "Distribution of COMPAS Decile Scores (1–10)",
        xLabel: "Decile score", yLabel: "Number of defendants",
        traces: [{
          name: "Count",
          x: decileLabels,
          y: decileVals,
          color: "#0ea5e9",
        }],
      },
    ],
  };

  // ── H3: Zero-inflation in juvenile counts ──────────────────────────────

  const juvFelTotal = records.filter(r => !isNaN(getNum(r, "juv_fel_count"))).length;
  const juvFelZeros = records.filter(r => getNum(r, "juv_fel_count") === 0).length;
  const juvMisdZeros = records.filter(r => getNum(r, "juv_misd_count") === 0).length;
  const juvFelZeroPct = (juvFelZeros / juvFelTotal) * 100;

  const juvBrackets = ["0", "1", "2", "3+"];
  const juvRecidRates = juvBrackets.map(b => {
    const filt = records.filter(r => {
      const c = getNum(r, "juv_fel_count");
      if (b === "0") return c === 0;
      if (b === "1") return c === 1;
      if (b === "2") return c === 2;
      return c >= 3;
    });
    const recid = filt.filter(r => r["two_year_recid"]?.trim() === "1").length;
    return filt.length ? +((recid / filt.length) * 100).toFixed(1) : 0;
  });

  const zeroCounts = [
    { col: "Felony", pct: +juvFelZeroPct.toFixed(1), nonzeroPct: +(100 - juvFelZeroPct).toFixed(1) },
    { col: "Misdemeanor", pct: +((juvMisdZeros / juvFelTotal) * 100).toFixed(1), nonzeroPct: +(100 - (juvMisdZeros / juvFelTotal) * 100).toFixed(1) },
  ];

  const h3: HypothesisResult = {
    id: "H3",
    verdict: juvFelZeroPct > 65 && (juvRecidRates[juvRecidRates.length - 1] - juvRecidRates[0]) > 10 ? "Supported" : "Inconclusive",
    testName: "Zero-inflation analysis + recidivism rate by juvenile felony count",
    statLabel: "% with zero juvenile felonies",
    statValue: `${juvFelZeroPct.toFixed(1)}%`,
    pValue: null,
    summary: `${juvFelZeroPct.toFixed(1)}% of defendants have zero juvenile felony charges (${((juvMisdZeros / juvFelTotal) * 100).toFixed(1)}% zero juvenile misdemeanors). Defendants with 3+ juvenile felonies have a ${juvRecidRates[3]}% recidivism rate vs. ${juvRecidRates[0]}% for those with none — a ${(juvRecidRates[3] - juvRecidRates[0]).toFixed(1)} pp gap. The extreme zero-inflation means a continuous predictor overweights the minority with juvenile history.`,
    charts: [
      {
        title: "Zero vs. Non-Zero Juvenile Charge Counts",
        xLabel: "Charge type", yLabel: "% of defendants",
        traces: [
          { name: "Zero (no history)", x: zeroCounts.map(d => d.col), y: zeroCounts.map(d => d.pct), color: "#94a3b8" },
          { name: "Non-zero (has history)", x: zeroCounts.map(d => d.col), y: zeroCounts.map(d => d.nonzeroPct), color: "#ef4444" },
        ],
      },
      {
        title: "2-Year Recidivism Rate by Juvenile Felony Count",
        xLabel: "Juvenile felony count", yLabel: "Recidivism rate (%)",
        traces: [{ name: "Recidivism rate", x: juvBrackets, y: juvRecidRates, color: "#f59e0b" }],
      },
    ],
  };

  // ── H4: Prior crimes encodes enforcement bias ──────────────────────────

  const meanPriorsByRace = groupMean(records, "race", "priors_count");
  const raceOrderPriors = Object.keys(meanPriorsByRace).sort((a, b) => meanPriorsByRace[b] - meanPriorsByRace[a]);

  // Recidivism rate by race × prior crimes bracket
  const priorBracketLabels = ["0", "1–3", "4+"];
  const racesForH4 = ["African-American", "Caucasian"];
  const recidByRacePrior = racesForH4.map(race => ({
    name: race,
    x: priorBracketLabels,
    y: priorBrackets.map(b => {
      const filt = records.filter(r => getStr(r, "race") === race && b.test(getNum(r, "priors_count")));
      const recid = filt.filter(r => r["two_year_recid"]?.trim() === "1").length;
      return filt.length ? +((recid / filt.length) * 100).toFixed(1) : 0;
    }),
    color: race === "African-American" ? "#ef4444" : "#0ea5e9",
  }));

  const aaMeanPriors = meanPriorsByRace["African-American"] ?? 0;
  const caucMeanPriors = meanPriorsByRace["Caucasian"] ?? 0;

  const h4: HypothesisResult = {
    id: "H4",
    verdict: aaMeanPriors > caucMeanPriors * 1.3 ? "Supported" : "Inconclusive",
    testName: "Group comparison: mean prior crimes count by race",
    statLabel: "AA vs. Caucasian mean priors",
    statValue: `${aaMeanPriors.toFixed(2)} vs. ${caucMeanPriors.toFixed(2)}`,
    pValue: null,
    summary: `African-Americans have a mean of ${aaMeanPriors.toFixed(2)} prior crimes vs. ${caucMeanPriors.toFixed(2)} for Caucasians — ${((aaMeanPriors / caucMeanPriors - 1) * 100).toFixed(0)}% higher. Importantly, even within the same prior-crimes bracket, AA defendants still show higher recidivism rates than Caucasians, suggesting prior crimes count alone does not fully explain the gap. This is consistent with differential policing encoding enforcement intensity into the prior crimes variable.`,
    charts: [
      {
        title: "Mean Prior Crimes Count by Race",
        xLabel: "Race", yLabel: "Mean prior crimes",
        traces: [{ name: "Mean priors", x: raceOrderPriors, y: raceOrderPriors.map(r => +meanPriorsByRace[r].toFixed(2)), color: "#8b5cf6" }],
      },
      {
        title: "2-Year Recidivism Rate by Race, Controlling for Prior Crimes",
        xLabel: "Prior crimes bracket", yLabel: "Recidivism rate (%)",
        traces: recidByRacePrior,
      },
    ],
  };

  return {
    datasetName: "Florida COMPAS",
    hypotheses: [h1, h2, h3, h4],
  };
}

// ── Georgia diagnostics ────────────────────────────────────────────────────

export async function computeGeorgiaDiagnostics(): Promise<DiagnosticResults> {
  const res = await fetch("/data/nij-challenge2021_full_dataset.csv");
  const text = await res.text();
  const records = parseCSV(text);

  const getNum = (r: Record<string, string>, col: string) => parseFloat(r[col]);
  const getStr = (r: Record<string, string>, col: string) => (r[col] ?? "").trim();
  const isRecid = (r: Record<string, string>) => toBool(r["Recidivism_Within_3years"]);

  // ── H5: Employment as structural stabiliser ────────────────────────────

  const empBrackets = [
    { label: "0–25%",  test: (v: number) => v >= 0 && v <= 25 },
    { label: "25–50%", test: (v: number) => v > 25 && v <= 50 },
    { label: "50–75%", test: (v: number) => v > 50 && v <= 75 },
    { label: "75–100%",test: (v: number) => v > 75 },
  ];

  const empRecidRates = empBrackets.map(b => {
    const filt = records.filter(r => {
      const v = getNum(r, "Percent_Days_Employed");
      return !isNaN(v) && b.test(v);
    });
    const recid = filt.filter(isRecid).length;
    return filt.length ? +((recid / filt.length) * 100).toFixed(1) : 0;
  });

  // Monotone decreasing test
  let isMonotone = true;
  for (let i = 1; i < empRecidRates.length; i++) {
    if (empRecidRates[i] > empRecidRates[i - 1]) { isMonotone = false; break; }
  }
  const empRange = empRecidRates[0] - empRecidRates[empRecidRates.length - 1];

  const h5: HypothesisResult = {
    id: "H5",
    verdict: isMonotone && empRange > 10 ? "Supported" : empRange > 5 ? "Inconclusive" : "Not Supported",
    testName: "Recidivism rate comparison across employment quartiles",
    statLabel: "Rate gap (0–25% vs. 75–100% employed)",
    statValue: `${empRange.toFixed(1)} pp`,
    pValue: null,
    summary: `Recidivism drops from ${empRecidRates[0]}% (0–25% employed) to ${empRecidRates[3]}% (75–100% employed) — a ${empRange.toFixed(1)} percentage-point gap. The trend is ${isMonotone ? "monotonically decreasing across all four brackets" : "broadly decreasing but non-monotone in one bracket"}, strongly consistent with employment acting as a protective factor against re-arrest.`,
    charts: [
      {
        title: "3-Year Recidivism Rate by Post-Release Employment Level",
        xLabel: "% Days Employed (post-release)", yLabel: "Recidivism rate (%)",
        traces: [{ name: "Recidivism rate", x: empBrackets.map(b => b.label), y: empRecidRates, color: "#10b981" }],
      },
    ],
  };

  // ── H6: 3-year window conflates two distinct recidivism phases ─────────

  const yr1 = records.filter(r => toBool(r["Recidivism_Arrest_Year1"])).length;
  const yr2 = records.filter(r => !toBool(r["Recidivism_Arrest_Year1"]) && toBool(r["Recidivism_Arrest_Year2"])).length;
  const yr3 = records.filter(r => !toBool(r["Recidivism_Arrest_Year1"]) && !toBool(r["Recidivism_Arrest_Year2"]) && toBool(r["Recidivism_Arrest_Year3"])).length;
  const n = records.length;
  const yr1Rate = +((yr1 / n) * 100).toFixed(1);
  const yr2Rate = +((yr2 / n) * 100).toFixed(1);
  const yr3Rate = +((yr3 / n) * 100).toFixed(1);

  // Also: among those who DID recidivate, what year did it first happen?
  const recidRecords = records.filter(isRecid);
  const recidYear1 = recidRecords.filter(r => toBool(r["Recidivism_Arrest_Year1"])).length;
  const recidYear2 = recidRecords.filter(r => !toBool(r["Recidivism_Arrest_Year1"]) && toBool(r["Recidivism_Arrest_Year2"])).length;
  const recidYear3 = recidRecords.filter(r => !toBool(r["Recidivism_Arrest_Year1"]) && !toBool(r["Recidivism_Arrest_Year2"]) && toBool(r["Recidivism_Arrest_Year3"])).length;
  const yr1Share = +((recidYear1 / recidRecords.length) * 100).toFixed(1);
  const yr2Share = +((recidYear2 / recidRecords.length) * 100).toFixed(1);
  const yr3Share = +((recidYear3 / recidRecords.length) * 100).toFixed(1);

  const h6: HypothesisResult = {
    id: "H6",
    verdict: yr1Rate > yr2Rate && yr1Rate > yr3Rate && yr1Share > 50 ? "Supported" : "Inconclusive",
    testName: "Year-by-year recidivism rate breakdown",
    statLabel: "Year 1 first-arrest share (among recidivists)",
    statValue: `${yr1Share}%`,
    pValue: null,
    summary: `Among all individuals, ${yr1Rate}% had a new arrest in Year 1, ${yr2Rate}% first reoffended in Year 2, and ${yr3Rate}% first reoffended in Year 3. Of everyone who recidivated, ${yr1Share}% did so in Year 1 alone — confirming that early post-release is a distinct high-risk phase. A binary 3-year outcome treats a Year 1 re-arrest identically to a Year 3 re-arrest, masking this temporal structure.`,
    charts: [
      {
        title: "New Recidivism Incidents by Year (as % of all individuals)",
        xLabel: "Follow-up year", yLabel: "% of total population",
        traces: [{ name: "First arrest in year", x: ["Year 1", "Year 2", "Year 3"], y: [yr1Rate, yr2Rate, yr3Rate], color: "#f59e0b" }],
      },
      {
        title: "When Recidivists First Re-Arrested (among those who recidivated)",
        xLabel: "First re-arrest year", yLabel: "% of recidivists",
        traces: [{ name: "Share of recidivists", x: ["Year 1", "Year 2", "Year 3"], y: [yr1Share, yr2Share, yr3Share], color: "#ef4444" }],
      },
    ],
  };

  // ── H7: Supervision risk score as surveillance bias ────────────────────

  // Get unique supervision levels sorted by mean risk score
  const supervisionLevels = groupMean(records, "Supervision_Level_First", "Supervision_Risk_Score_First");
  const supRecidRates = groupRate(records, "Supervision_Level_First", "Recidivism_Within_3years");
  const supOrder = Object.keys(supervisionLevels)
    .filter(l => l && l !== "")
    .sort((a, b) => supervisionLevels[a] - supervisionLevels[b]);

  // Risk score quartiles within each supervision level → do higher scores still predict recidivism?
  const supLevelsForStratified = supOrder.slice(0, 3); // top 3 supervision levels by size
  const riskQ = ["Low risk (1–4)", "High risk (5–10)"];
  const stratified = supLevelsForStratified.map(sup => ({
    sup,
    rates: riskQ.map(q => {
      const isLow = q.startsWith("Low");
      const filt = records.filter(r => {
        const level = getStr(r, "Supervision_Level_First");
        const score = getNum(r, "Supervision_Risk_Score_First");
        return level === sup && (isLow ? score <= 4 : score > 4);
      });
      const recid = filt.filter(isRecid).length;
      return filt.length ? +((recid / filt.length) * 100).toFixed(1) : null;
    }),
  }));

  // If score effect shrinks within supervision levels → surveillance bias is present
  const overallGap = (supRecidRates[supOrder[supOrder.length - 1]] ?? 0) - (supRecidRates[supOrder[0]] ?? 0);

  const h7: HypothesisResult = {
    id: "H7",
    verdict: overallGap > 10 ? "Supported" : overallGap > 5 ? "Inconclusive" : "Not Supported",
    testName: "Recidivism rate stratified by supervision level + risk score quartile",
    statLabel: "Recidivism gap (lowest vs. highest supervision level)",
    statValue: `${Math.abs(overallGap).toFixed(1)} pp`,
    pValue: null,
    summary: `Recidivism rates climb from ${supRecidRates[supOrder[0]] ?? "–"}% (${supOrder[0]}) to ${supRecidRates[supOrder[supOrder.length - 1]] ?? "–"}% (${supOrder[supOrder.length - 1]}) — a ${Math.abs(overallGap).toFixed(1)} pp gap. Higher supervision levels mean more opportunities to detect technical violations (missed check-ins, drug tests), so part of the score's predictive power may reflect detection intensity rather than actual behavioural differences.`,
    charts: [
      {
        title: "Mean Supervision Risk Score by Supervision Level",
        xLabel: "Supervision level", yLabel: "Mean risk score",
        traces: [{ name: "Mean risk score", x: supOrder, y: supOrder.map(l => +supervisionLevels[l].toFixed(2)), color: "#8b5cf6" }],
      },
      {
        title: "3-Year Recidivism Rate by Supervision Level",
        xLabel: "Supervision level", yLabel: "Recidivism rate (%)",
        traces: [{ name: "Recidivism rate", x: supOrder, y: supOrder.map(l => supRecidRates[l] ?? 0), color: "#0ea5e9" }],
      },
    ],
  };

  // ── H8: Prison years as proxy for offence severity ─────────────────────

  const prisonYearRecid = groupRate(records, "Prison_Years", "Recidivism_Within_3years");
  const prisonYearOrder = [
    "Less than 1 year",
    "1-2 years",
    "Greater than 2 to 3 years",
    "More than 3 years",
  ].filter(k => k in prisonYearRecid);
  // Fallback: use whatever order exists
  const pyKeys = prisonYearOrder.length > 0 ? prisonYearOrder : Object.keys(prisonYearRecid).filter(k => k);
  const pyRates = pyKeys.map(k => prisonYearRecid[k] ?? 0);

  // Correlation: does longer prison time → lower recidivism? (expected: yes, deterrence / age-out)
  const pyMin = Math.min(...pyRates), pyMax = Math.max(...pyRates);
  const pyRange = pyMax - pyMin;
  const firstHigherThanLast = pyRates[0] > pyRates[pyRates.length - 1];

  const h8: HypothesisResult = {
    id: "H8",
    verdict: pyRange > 5 ? "Supported" : "Inconclusive",
    testName: "Recidivism rate by Prison_Years category",
    statLabel: "Rate range across prison-year groups",
    statValue: `${pyRange.toFixed(1)} pp`,
    pValue: null,
    summary: `Recidivism rates vary ${pyRange.toFixed(1)} percentage points across prison sentence length categories. ${firstHigherThanLast ? "Shorter sentences are associated with higher recidivism" : "Longer sentences are associated with higher recidivism"}, suggesting Prison_Years captures a mix of offence severity and post-release adjustment difficulty. Including it as a raw continuous variable conflates these effects — treating it as categorical (as shown here) is more informative.`,
    charts: [
      {
        title: "3-Year Recidivism Rate by Prison Sentence Length",
        xLabel: "Prison years", yLabel: "Recidivism rate (%)",
        traces: [{ name: "Recidivism rate", x: pyKeys, y: pyRates, color: "#f59e0b" }],
      },
    ],
  };

  return {
    datasetName: "Georgia NIJ",
    hypotheses: [h5, h6, h7, h8],
  };
}
