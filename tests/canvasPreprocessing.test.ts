import { normalizeCanvasImage } from "@/lib/canvasPreprocessing";

function buildImageData(width: number, height: number, inkCells: Array<[number, number]>) {
  const data = new Uint8ClampedArray(width * height * 4);

  for (let index = 0; index < data.length; index += 4) {
    data[index + 3] = 255;
  }

  for (const [x, y] of inkCells) {
    const offset = (y * width + x) * 4;
    data[offset] = 255;
    data[offset + 1] = 255;
    data[offset + 2] = 255;
    data[offset + 3] = 255;
  }

  return { width, height, data };
}

function centerOfMass(matrix: number[][]) {
  let total = 0;
  let weightedX = 0;
  let weightedY = 0;

  for (let y = 0; y < matrix.length; y += 1) {
    for (let x = 0; x < matrix[y].length; x += 1) {
      const value = matrix[y][x];
      total += value;
      weightedX += x * value;
      weightedY += y * value;
    }
  }

  return {
    x: weightedX / total,
    y: weightedY / total
  };
}

describe("normalizeCanvasImage", () => {
  it("returns null when the canvas is blank", () => {
    const imageData = buildImageData(12, 12, []);
    expect(normalizeCanvasImage(imageData)).toBeNull();
  });

  it("crops, centers, and rescales ink into a 28x28 matrix", () => {
    const imageData = buildImageData(20, 20, [
      [2, 10],
      [3, 10],
      [4, 10],
      [4, 11],
      [4, 12],
      [4, 13]
    ]);

    const result = normalizeCanvasImage(imageData);

    expect(result).not.toBeNull();
    expect(result).toHaveLength(28);
    expect(result?.[0]).toHaveLength(28);

    const flattened = result?.flat() ?? [];
    expect(Math.max(...flattened)).toBeGreaterThan(0.2);
    expect(Math.min(...flattened)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...flattened)).toBeLessThanOrEqual(1);

    const center = centerOfMass(result ?? []);
    expect(center.x).toBeGreaterThan(9);
    expect(center.x).toBeLessThan(18);
    expect(center.y).toBeGreaterThan(9);
    expect(center.y).toBeLessThan(18);
  });
});

