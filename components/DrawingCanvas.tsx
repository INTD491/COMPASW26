"use client";

import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import { canvasToMnistInput } from "@/lib/canvasPreprocessing";
import type { PredictionResponse } from "@/lib/types";

import { PredictionChart } from "@/components/PredictionChart";

const CANVAS_SIZE = 280;
const LINE_WIDTH = 18;

type Point = {
  x: number;
  y: number;
};

function formatConfidence(probabilities: number[] | null, digit: number | null): string {
  if (!probabilities || digit === null) {
    return "";
  }

  return `${(probabilities[digit] * 100).toFixed(1)}% confidence`;
}

export function DrawingCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDrawingRef = useRef(false);
  const lastPointRef = useRef<Point | null>(null);

  const [prediction, setPrediction] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    context.fillStyle = "#000000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "#ffffff";
    context.lineWidth = LINE_WIDTH;
  }, []);

  const resetPrediction = () => {
    setPrediction(null);
    setProbabilities(null);
    setError(null);
  };

  const getPoint = (event: ReactPointerEvent<HTMLCanvasElement>): Point => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return { x: 0, y: 0 };
    }

    const bounds = canvas.getBoundingClientRect();
    const scaleX = canvas.width / bounds.width;
    const scaleY = canvas.height / bounds.height;

    return {
      x: (event.clientX - bounds.left) * scaleX,
      y: (event.clientY - bounds.top) * scaleY
    };
  };

  const drawSegment = (start: Point, end: Point) => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = getPoint(event);
    isDrawingRef.current = true;
    lastPointRef.current = point;
    drawSegment(point, point);
    event.currentTarget.setPointerCapture(event.pointerId);
    resetPrediction();
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current || !lastPointRef.current) {
      return;
    }

    const point = getPoint(event);
    drawSegment(lastPointRef.current, point);
    lastPointRef.current = point;
  };

  const finishStroke = () => {
    isDrawingRef.current = false;
    lastPointRef.current = null;
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    context.fillStyle = "#000000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    resetPrediction();
  };

  const handlePredict = async () => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const image = canvasToMnistInput(canvas);

    if (!image) {
      setError("Draw a digit before running the prediction.");
      setPrediction(null);
      setProbabilities(null);
      return;
    }

    setIsPredicting(true);
    setError(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ image })
      });

      const payload = (await response.json()) as PredictionResponse | { error: string };

      if (!response.ok || !("predicted_digit" in payload) || !("probabilities" in payload)) {
        throw new Error("error" in payload ? payload.error : "Prediction failed.");
      }

      setPrediction(payload.predicted_digit);
      setProbabilities(payload.probabilities);
    } catch (requestError) {
      setPrediction(null);
      setProbabilities(null);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Prediction failed. Please try again."
      );
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="predict-grid">
      <div className="canvas-panel">
        <div className="canvas-shell">
          <canvas
            ref={canvasRef}
            aria-label="Digit drawing canvas"
            className="drawing-canvas"
            height={CANVAS_SIZE}
            onPointerCancel={finishStroke}
            onPointerDown={handlePointerDown}
            onPointerLeave={finishStroke}
            onPointerMove={handlePointerMove}
            onPointerUp={finishStroke}
            width={CANVAS_SIZE}
          />
        </div>
        <p className="helper-text">
          Draw a single digit with a white stroke on the black canvas. The frontend crops, centers,
          rescales, and normalizes the image to the 28x28 MNIST format before calling the backend.
        </p>
        <div className="button-row">
          <button className="button button--primary" disabled={isPredicting} onClick={handlePredict}>
            {isPredicting ? "Predicting..." : "Predict"}
          </button>
          <button className="button button--secondary" disabled={isPredicting} onClick={clearCanvas}>
            Clear
          </button>
        </div>
        {error ? <p className="status-error">{error}</p> : null}
      </div>
      <div className="section-card">
        <div className="section-card__body">
          <div className="section-card__header">
            <h3>Prediction Output</h3>
            <p>Backend inference runs in Python with pure NumPy weights exported from PyTorch.</p>
          </div>
          {prediction !== null ? (
            <>
              <div className="prediction-badge">
                <span>Predicted digit</span>
                <strong>{prediction}</strong>
              </div>
              <p className="prediction-meta">{formatConfidence(probabilities, prediction)}</p>
            </>
          ) : (
            <div className="canvas-empty">
              Draw a digit and click Predict to compare the class scores across all ten digits.
            </div>
          )}
          <div style={{ marginTop: "16px" }}>
            <PredictionChart probabilities={probabilities} />
          </div>
        </div>
      </div>
    </div>
  );
}
