"""
POST /degradation — LSTM degradation simulation endpoint
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import DegradationRequest, DegradationResponse, CyclePoint
from backend.models.db_models import DegradationResult
from backend.ml.lstm_degradation import simulate_degradation
from backend.ml.predictor import predict_properties, parse_formula

router = APIRouter(prefix="/degradation", tags=["Degradation"])


@router.post("", response_model=DegradationResponse)
async def degradation(request: DegradationRequest, db: Session = Depends(get_db)):
    """
    Simulate battery capacity fade and voltage decay over charge/discharge cycles.
    Uses LSTM model (physics-backed fallback for untrained model).
    """
    # Get stability estimate for the formula
    try:
        comp = parse_formula(request.formula)
        props = predict_properties(request.formula, composition=comp)
        stability = props["structural_stability"]
    except Exception:
        stability = 0.05  # default moderate stability

    try:
        result = simulate_degradation(
            formula=request.formula,
            initial_capacity=request.initial_capacity,
            total_cycles=request.total_cycles,
            temperature_c=request.temperature_c,
            c_rate=request.c_rate,
            stability=stability,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Degradation simulation failed: {str(e)}")

    # Persist to DB
    try:
        db_deg = DegradationResult(
            formula=request.formula,
            total_cycles=request.total_cycles,
            initial_capacity=request.initial_capacity,
            final_capacity=result["final_capacity"],
            capacity_retention=result["capacity_retention"],
            voltage_decay=result["voltage_decay"],
            cycle_life_predicted=result["cycle_life_predicted"],
            degradation_index=result["degradation_index"],
            curve_data_json=result["curve_data"],
            model_type=result["model_type"],
        )
        db.add(db_deg)
        db.commit()
        db.refresh(db_deg)
        result["id"] = db_deg.id
    except Exception:
        pass

    return DegradationResponse(
        id=result.get("id"),
        formula=result["formula"],
        total_cycles=result["total_cycles"],
        initial_capacity=result["initial_capacity"],
        final_capacity=result["final_capacity"],
        capacity_retention=result["capacity_retention"],
        voltage_decay=result["voltage_decay"],
        cycle_life_predicted=result["cycle_life_predicted"],
        degradation_index=result["degradation_index"],
        curve_data=[CyclePoint(**p) for p in result["curve_data"]],
        model_type=result["model_type"],
        summary=result["summary"],
    )
