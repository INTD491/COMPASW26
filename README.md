# COMPAS Dashboard: Interpretable & Fair Recidivism Prediction

An interactive web dashboard for evaluating fairness, interpretability, and generalization in machine learning models for recidivism prediction. Built for INTD 491: Cloud Deployment (University of Alberta, Winter 2026).

## Features

- **Home Dashboard**: Overview of dataset metrics and model performance
- **Prediction Tool**: Interactive risk assessment with multiple model comparison
- **Fairness Analysis**: Fairness metrics, demographic parity, and debiasing methods
- **Model Comparison**: Performance cards, ROC curves, confusion matrices
- **About**: Project information, research questions, and references

## Stack

- Next.js 14 App Router with TypeScript
- React and Plotly for interactive visualizations
- Python API for mock recidivism predictions
- Vercel Cloud Deployment
- COMPAS Florida and NIJ Georgia datasets

## Folder Structure

```text
app/                  Next.js pages (home, prediction, fairness, models, about)
components/           Charts, navigation, and reusable UI components
api/                  Vercel Python prediction endpoints
lib/                  TypeScript helpers and types
public/               Static assets
datasets/             COMPAS and NIJ data files
tests/                Frontend and backend testing
```

## Local Setup

1. Install Node dependencies:

   ```bash
   npm install
   ```

2. Install Python runtime dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```

## Development Server

Run the development server:

```bash
npm run dev:next
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## Deployment

Deploy to Vercel with Git integration:

```bash
npm run build
npm start
```

## Testing

Run all tests:

```bash
npm test
```

Run frontend tests only:

```bash
npm run test:frontend
```

Run Python tests only:

```bash
npm run test:python
```

```bash
python3 scripts/preprocess_mnist.py
```

Outputs:

- `public/data/mnist_stats.json`
- `public/data/mnist_preview.csv`

## Train And Export The CNN

The local training script uses a small CNN:

- `Conv2d(1, 8, 3, padding=1)`
- `ReLU`
- `MaxPool2d(2)`
- `Conv2d(8, 16, 3, padding=1)`
- `ReLU`
- `MaxPool2d(2)`
- `Linear(16 * 7 * 7, 10)`

Run:

```bash
python3 scripts/train_and_export.py
```

This writes `model/mnist_cnn_weights.npz`, which the deployed API loads with NumPy only.

## Run The App Locally

Use Vercel's local development server so both the Next.js frontend and the Python prediction function are available together:

```bash
npm run dev
```

Open the printed local URL in your browser. If you only want to inspect the Next.js frontend without the Python function, use:

```bash
npm run dev:next
```

## Prediction API Contract

`POST /api/predict`

Request body:

```json
{
  "image": [[0.0, 0.1], [0.3, 0.9]]
}
```

The real payload must contain a normalized 28x28 matrix.

Response body:

```json
{
  "predicted_digit": 7,
  "probabilities": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.55, 0.09, 0.08]
}
```

## Canvas Preprocessing Notes

- The canvas uses a black background with a white stroke to match MNIST's digit-on-dark-background appearance.
- Before sending the request, the frontend reads the canvas pixels, finds the non-empty bounding box, pads it to a square, rescales it to 28x28, and normalizes values to the `[0, 1]` range.
- This crop-and-center step helps the backend handle smaller or off-center drawings more reliably than a naive full-canvas resize.

## Why The Deployed API Avoids PyTorch

PyTorch is useful for local training, but it is much heavier than needed for a single-sample MNIST demo. The deployed API only needs NumPy plus the exported `.npz` weights, which keeps the Vercel runtime smaller and the inference path easy to explain in class.

## Tests

Run both the frontend and Python smoke tests:

```bash
npm test
```

Run the production build check:

```bash
npm run build
```

## Deploy To Vercel

1. Push the repo to GitHub.
2. Import the project into Vercel.
3. Ensure `requirements.txt` is present so the Python function installs NumPy.
4. Deploy normally. The frontend assets are served by Next.js and the prediction endpoint is served from `api/predict.py`.
