from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


def predict_recidivism(
    age: int,
    sex: str,
    race: str,
    prior_crimes: int,
    juvenile_felonies: int,
    charge_degree: str,
    model: str
) -> tuple[float, str]:
    """Predict recidivism risk for an individual."""
    base_score = 0.45

    if age < 25:
        base_score += 0.15
    elif age > 60:
        base_score -= 0.1
    else:
        base_score -= (age - 25) * 0.003

    base_score += min(prior_crimes * 0.03, 0.25)
    base_score += juvenile_felonies * 0.05

    if race == "African American":
        base_score += 0.08
    elif race == "Hispanic":
        base_score += 0.03

    if charge_degree == "Felony":
        base_score += 0.05

    model_adjustment = {
        "Logistic Regression": -0.02,
        "Random Forest": 0.0,
        "Decision Tree": -0.01,
        "XGBoost + Debiasing": -0.05
    }.get(model, 0.0)

    base_score += model_adjustment
    risk_score = max(0.0, min(1.0, base_score))

    return risk_score, model


def parse_recidivism_input(payload: object) -> dict:
    """Validate and parse COMPAS prediction request."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    required_fields = {
        "age": int,
        "sex": str,
        "race": str,
        "prior_crimes": int,
        "juvenile_felonies": int,
        "charge_degree": str,
        "model": str
    }

    parsed = {}
    for field, field_type in required_fields.items():
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")

        value = payload[field]
        if not isinstance(value, field_type):
            type_name = field_type.__name__
            val_type_name = type(value).__name__
            raise ValueError(
                f"Field {field} must be {type_name}, got {val_type_name}"
            )

        parsed[field] = value

    if not 18 <= parsed["age"] <= 80:
        raise ValueError("Age must be between 18 and 80")
    if parsed["sex"] not in ["Male", "Female"]:
        raise ValueError("Sex must be Male or Female")

    races = ["African American", "Caucasian", "Hispanic", "Other"]
    if parsed["race"] not in races:
        raise ValueError("Invalid race value")

    if not 0 <= parsed["prior_crimes"] <= 40:
        raise ValueError("Prior crimes must be between 0 and 40")
    if not 0 <= parsed["juvenile_felonies"] <= 10:
        raise ValueError("Juvenile felonies must be between 0 and 10")

    charge_vals = ["Felony", "Misdemeanor"]
    if parsed["charge_degree"] not in charge_vals:
        raise ValueError("Charge degree must be Felony or Misdemeanor")

    return parsed


def build_prediction_response(body: bytes) -> tuple[int, dict[str, object]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return 400, {"error": "Request body must be valid JSON."}

    try:
        input_data = parse_recidivism_input(payload)
        risk_score, model_used = predict_recidivism(**input_data)
    except ValueError as error:
        return 400, {"error": str(error)}

    risk_level = (
        "high"
        if risk_score > 0.6
        else "medium"
        if risk_score > 0.3
        else "low"
    )

    return 200, {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "model": model_used,
        "input": input_data
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(
        self, status_code: int, payload: dict[str, object]
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        status_code, payload = build_prediction_response(body)
        self._send_json(status_code, payload)

    def do_GET(self) -> None:  # noqa: N802
        error_msg = "Use POST /api/predict with COMPAS data payload."
        self._send_json(405, {"error": error_msg})
