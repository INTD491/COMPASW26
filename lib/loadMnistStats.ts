import { readFile } from "node:fs/promises";
import path from "node:path";

import type { MnistStats } from "@/lib/types";

const statsPath = path.join(process.cwd(), "public", "data", "mnist_stats.json");

export async function loadMnistStats(): Promise<MnistStats> {
  const raw = await readFile(statsPath, "utf8");
  return JSON.parse(raw) as MnistStats;
}

