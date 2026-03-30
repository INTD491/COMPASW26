import type { ImageDataLike } from "@/lib/types";

const INK_THRESHOLD = 12 / 255;

function createMatrix(size: number): number[][] {
  return Array.from({ length: size }, () => Array<number>(size).fill(0));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function rgbaToGrayscale(imageData: ImageDataLike): number[][] {
  const pixels: number[][] = [];

  for (let y = 0; y < imageData.height; y += 1) {
    const row: number[] = [];

    for (let x = 0; x < imageData.width; x += 1) {
      const offset = (y * imageData.width + x) * 4;
      row.push(Number(imageData.data[offset] ?? 0) / 255);
    }

    pixels.push(row);
  }

  return pixels;
}

function findBoundingBox(source: number[][]) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let y = 0; y < source.length; y += 1) {
    for (let x = 0; x < source[y].length; x += 1) {
      if (source[y][x] <= INK_THRESHOLD) {
        continue;
      }

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (!Number.isFinite(minX)) {
    return null;
  }

  return { minX, minY, maxX, maxY };
}

function resizeMatrix(source: number[][], targetSize: number): number[][] {
  const sourceHeight = source.length;
  const sourceWidth = source[0]?.length ?? 0;
  const target = Array.from({ length: targetSize }, () => Array<number>(targetSize).fill(0));

  if (sourceHeight === 0 || sourceWidth === 0) {
    return target;
  }

  const xRatio = sourceWidth / targetSize;
  const yRatio = sourceHeight / targetSize;

  for (let y = 0; y < targetSize; y += 1) {
    const sourceY = (y + 0.5) * yRatio - 0.5;
    const y0 = clamp(Math.floor(sourceY), 0, sourceHeight - 1);
    const y1 = clamp(y0 + 1, 0, sourceHeight - 1);
    const wy = sourceY - y0;

    for (let x = 0; x < targetSize; x += 1) {
      const sourceX = (x + 0.5) * xRatio - 0.5;
      const x0 = clamp(Math.floor(sourceX), 0, sourceWidth - 1);
      const x1 = clamp(x0 + 1, 0, sourceWidth - 1);
      const wx = sourceX - x0;

      const top = source[y0][x0] * (1 - wx) + source[y0][x1] * wx;
      const bottom = source[y1][x0] * (1 - wx) + source[y1][x1] * wx;
      target[y][x] = top * (1 - wy) + bottom * wy;
    }
  }

  return target;
}

export function normalizeCanvasImage(imageData: ImageDataLike): number[][] | null {
  const grayscale = rgbaToGrayscale(imageData);
  const bounds = findBoundingBox(grayscale);

  if (!bounds) {
    return null;
  }

  const cropWidth = bounds.maxX - bounds.minX + 1;
  const cropHeight = bounds.maxY - bounds.minY + 1;
  const margin = Math.max(2, Math.round(Math.max(cropWidth, cropHeight) * 0.18));
  const squareSize = Math.max(cropWidth, cropHeight) + margin * 2;
  const centered = createMatrix(squareSize);
  const offsetX = Math.floor((squareSize - cropWidth) / 2);
  const offsetY = Math.floor((squareSize - cropHeight) / 2);

  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      centered[offsetY + y][offsetX + x] = grayscale[bounds.minY + y][bounds.minX + x];
    }
  }

  return resizeMatrix(centered, 28);
}

export function canvasToMnistInput(canvas: HTMLCanvasElement): number[][] | null {
  const context = canvas.getContext("2d");

  if (!context) {
    return null;
  }

  return normalizeCanvasImage(context.getImageData(0, 0, canvas.width, canvas.height));
}

