"""
ChurnIQ - FastAPI Backend
Endpoints: POST /predict, POST /batch_predict, POST /chat

Run from project root:
    uvicorn app.main:app --reload

API docs available at:
    http://127.0.0.1:8000/docs
"""

import io
import os
import sys
import pandas as pd
from contextlib import asynccontextmanager

# ─────────────────────────────────────────────
# Ensure project root is on sys.path so `src` is importable
# when uvicorn is launched from any directory.
# ─────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.predict import predict_churn, predict_batch
from src.agent import build_agent, run_agent


# ─────────────────────────────────────────────
# App state: hold the agent across requests
# ─────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the LangChain agent once at startup, reuse on every /chat call."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            _state["agent"] = build_agent()
            print("✅  ChurnIQ agent loaded.")
        except Exception as e:
            print(f"⚠️  Agent failed to load: {e}. /chat endpoint will be unavailable.")
            _state["agent"] = None
    else:
        print("⚠️  GROQ_API_KEY not set. /chat endpoint will be unavailable.")
        _state["agent"] = None
    yield
    _state.clear()


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="ChurnIQ API",
    description="SaaS churn prediction and AI analyst API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────
class CustomerInput(BaseModel):
    features: dict  # {column_name: value} — all 28 feature columns required

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "tenure": 2,
                    "MonthlyCharges": 79.85,
                    "TotalCharges": 159.70,
                    "SeniorCitizen": 0,
                    "Partner": 0,
                    "Dependents": 0,
                    "PhoneService": 1,
                    "MultipleLines": 0,
                    "OnlineSecurity": 0,
                    "OnlineBackup": 0,
                    "DeviceProtection": 0,
                    "TechSupport": 0,
                    "StreamingTV": 0,
                    "StreamingMovies": 0,
                    "PaperlessBilling": 1,
                    "gender_Male": 1,
                    "InternetService_Fiber optic": 1,
                    "InternetService_No": 0,
                    "Contract_One year": 0,
                    "Contract_Two year": 0,
                    "PaymentMethod_Credit card (automatic)": 0,
                    "PaymentMethod_Electronic check": 1,
                    "PaymentMethod_Mailed check": 0,
                    "tenure_group": 0,
                    "avg_monthly_to_total_ratio": 0.499,
                    "services_count": 1,
                    "high_value_at_risk": 1,
                    "support_bundle": 0
                }
            }
        }
    }


class PredictResponse(BaseModel):
    churn_probability: float
    risk_level: str
    churn_prediction: int


class ChatInput(BaseModel):
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {"message": "Which customers are at highest risk of churning?"}
        }
    }


class ChatResponse(BaseModel):
    response: str


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "ChurnIQ API",
        "agent_loaded": _state.get("agent") is not None,
    }


# ─────────────────────────────────────────────
# POST /predict — single customer
# ─────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict churn probability for a single customer.

    Expects a JSON body with a 'features' dict containing all 28 model features.
    Returns churn_probability (0–1), risk_level (High/Medium/Low), churn_prediction (0/1).
    """
    try:
        result = predict_churn(customer.features)
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Missing feature: {e}. Check feature_columns.json for required fields.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ─────────────────────────────────────────────
# POST /batch_predict — CSV file upload
# ─────────────────────────────────────────────
@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Predict churn for all customers in an uploaded CSV file.

    Upload a CSV with the same columns as the training data (Churn column optional).
    Returns a CSV file with three columns added: churn_probability, risk_level, churn_prediction.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        # Drop Churn column if present (it's the target, not a feature)
        input_df = df.drop(columns=["Churn"], errors="ignore")
        result_df = predict_batch(input_df)
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required feature column: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return enriched CSV as a downloadable file
    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churniq_predictions.csv"},
    )


# ─────────────────────────────────────────────
# POST /chat — AI analyst agent
# ─────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["AI Analyst"])
def chat(body: ChatInput):
    """
    Ask the ChurnIQ AI analyst a question about your customers.

    The agent uses the trained model and customer data to answer in plain English.
    Requires GROQ_API_KEY to be set.
    """
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="AI analyst is unavailable. Ensure GROQ_API_KEY is set and restart the server.",
        )

    try:
        response = run_agent(agent, body.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return {"response": response}