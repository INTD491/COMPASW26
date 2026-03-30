import csv
import json
import unittest
from pathlib import Path

import numpy as np

from api.inference import load_weights, predict_image
from api.predict import build_prediction_response


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_preview_image() -> list[list[float]]:
    preview_path = PROJECT_ROOT / "public" / "data" / "mnist_preview.csv"

    with preview_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    pixels = [float(row[f"pixel_{index:03d}"]) / 255.0 for index in range(784)]
    return [pixels[index : index + 28] for index in range(0, 784, 28)]


class InferenceTests(unittest.TestCase):
    def test_weights_file_loads(self) -> None:
        weights = load_weights()
        self.assertIn("conv1_weight", weights)
        self.assertIn("fc_weight", weights)

    def test_valid_payload_returns_probabilities(self) -> None:
        image = load_preview_image()
        status_code, payload = build_prediction_response(
            json.dumps({"image": image}).encode("utf-8")
        )

        self.assertEqual(status_code, 200)
        probabilities = payload["probabilities"]
        self.assertEqual(len(probabilities), 10)
        self.assertTrue(all(np.isfinite(probabilities)))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=5)
        self.assertEqual(payload["predicted_digit"], int(np.argmax(probabilities)))

    def test_malformed_payload_returns_400(self) -> None:
        status_code, payload = build_prediction_response(b'{"image": [1, 2, 3]}')
        self.assertEqual(status_code, 400)
        self.assertIn("error", payload)

    def test_prediction_is_stable_for_known_sample(self) -> None:
        image = load_preview_image()
        first_digit, first_probabilities = predict_image(image)
        second_digit, second_probabilities = predict_image(image)

        self.assertEqual(first_digit, second_digit)
        np.testing.assert_allclose(first_probabilities, second_probabilities, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

