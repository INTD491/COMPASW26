import { NextRequest, NextResponse } from "next/server";

function predictRecidivism(
  age: number,
  sex: string,
  race: string,
  priorCrimes: number,
  juvenileFelonies: number,
  chargeDegree: string,
  model: string
): [number, string] {
  let baseScore = 0.45;

  if (age < 25) {
    baseScore += 0.15;
  } else if (age > 60) {
    baseScore -= 0.1;
  } else {
    baseScore -= (age - 25) * 0.003;
  }

  baseScore += Math.min(priorCrimes * 0.03, 0.25);
  baseScore += juvenileFelonies * 0.05;

  if (race === "African American") {
    baseScore += 0.08;
  } else if (race === "Hispanic") {
    baseScore += 0.03;
  }

  if (chargeDegree === "Felony") {
    baseScore += 0.05;
  }

  const modelAdjustment: { [key: string]: number } = {
    "Logistic Regression": -0.02,
    "Random Forest": 0.0,
    "Decision Tree": -0.01,
    "XGBoost + Debiasing": -0.05,
  };

  baseScore += modelAdjustment[model] || 0.0;

  const riskScore = Math.max(0.0, Math.min(1.0, baseScore));

  return [riskScore, model];
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    const required = [
      "age",
      "sex",
      "race",
      "prior_crimes",
      "juvenile_felonies",
      "charge_degree",
      "model",
    ];
    for (const field of required) {
      if (!(field in body)) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        );
      }
    }

    // Validate ranges
    if (body.age < 18 || body.age > 80) {
      return NextResponse.json(
        { error: "Age must be between 18 and 80" },
        { status: 400 }
      );
    }

    if (!["Male", "Female"].includes(body.sex)) {
      return NextResponse.json(
        { error: "Sex must be Male or Female" },
        { status: 400 }
      );
    }

    const validRaces = ["African American", "Caucasian", "Hispanic", "Other"];
    if (!validRaces.includes(body.race)) {
      return NextResponse.json(
        { error: "Invalid race value" },
        { status: 400 }
      );
    }

    if (
      body.prior_crimes < 0 ||
      body.prior_crimes > 40
    ) {
      return NextResponse.json(
        { error: "Prior crimes must be between 0 and 40" },
        { status: 400 }
      );
    }

    if (
      body.juvenile_felonies < 0 ||
      body.juvenile_felonies > 10
    ) {
      return NextResponse.json(
        { error: "Juvenile felonies must be between 0 and 10" },
        { status: 400 }
      );
    }

    if (!["Felony", "Misdemeanor"].includes(body.charge_degree)) {
      return NextResponse.json(
        { error: "Charge degree must be Felony or Misdemeanor" },
        { status: 400 }
      );
    }

    // Make prediction
    const [riskScore, modelUsed] = predictRecidivism(
      body.age,
      body.sex,
      body.race,
      body.prior_crimes,
      body.juvenile_felonies,
      body.charge_degree,
      body.model
    );

    const riskLevel =
      riskScore > 0.6 ? "high" : riskScore > 0.3 ? "medium" : "low";

    return NextResponse.json({
      risk_score: riskScore,
      risk_level: riskLevel,
      model: modelUsed,
      input: body,
    });
  } catch (error) {
    console.error("Prediction error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}
