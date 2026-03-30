from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "mnist_cnn_weights.npz"


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0).astype(np.float32, copy=False)


def max_pool2d(values: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
    channels, height, width = values.shape
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1
    pooled = np.zeros((channels, output_height, output_width), dtype=np.float32)

    for channel in range(channels):
        for y in range(output_height):
            for x in range(output_width):
                region = values[
                    channel,
                    y * stride : y * stride + kernel_size,
                    x * stride : x * stride + kernel_size,
                ]
                pooled[channel, y, x] = np.max(region)

    return pooled


def conv2d(values: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: int = 1) -> np.ndarray:
    in_channels, height, width = values.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    padded = np.pad(values, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
    output = np.zeros((out_channels, height, width), dtype=np.float32)

    for out_channel in range(out_channels):
        output[out_channel] = bias[out_channel]

        for in_channel in range(in_channels):
            kernel = weight[out_channel, in_channel]

            for y in range(height):
                for x in range(width):
                    region = padded[in_channel, y : y + kernel_height, x : x + kernel_width]
                    output[out_channel, y, x] += np.sum(region * kernel, dtype=np.float32)

    return output


def linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return weight @ values + bias


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted).astype(np.float32, copy=False)
    return exp_values / np.sum(exp_values, dtype=np.float32)


@lru_cache(maxsize=1)
def load_weights(model_path: str = str(MODEL_PATH)) -> dict[str, np.ndarray]:
    with np.load(model_path) as archive:
        return {name: archive[name].astype(np.float32) for name in archive.files}


def predict_probabilities(image: list[list[float]] | np.ndarray) -> np.ndarray:
    weights = load_weights()
    array = np.asarray(image, dtype=np.float32)

    if array.shape != (28, 28):
        raise ValueError("Image must be a 28x28 matrix.")

    features = array.reshape(1, 28, 28)
    hidden = conv2d(features, weights["conv1_weight"], weights["conv1_bias"], padding=1)
    hidden = relu(hidden)
    hidden = max_pool2d(hidden, kernel_size=2, stride=2)
    hidden = conv2d(hidden, weights["conv2_weight"], weights["conv2_bias"], padding=1)
    hidden = relu(hidden)
    hidden = max_pool2d(hidden, kernel_size=2, stride=2)
    logits = linear(hidden.reshape(-1), weights["fc_weight"], weights["fc_bias"])
    return softmax(logits)


def predict_image(image: list[list[float]] | np.ndarray) -> tuple[int, list[float]]:
    probabilities = predict_probabilities(image)
    return int(np.argmax(probabilities)), probabilities.astype(float).tolist()
