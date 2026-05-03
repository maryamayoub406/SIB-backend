"""
POST /predict — Property prediction endpoint
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import PredictionRequest, PredictionResponse, ValidationResult, SHAPEntry
from backend.models.db_models import Prediction
from backend.ml.predictor import predict_properties

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse)
async def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Predict battery properties for a given formula or composition.
    Returns 8 key properties, SHAP values, validation, and explanation.
    """
    try:
        result = predict_properties(
            formula=request.formula,
            composition=request.composition,
            declared_elements=request.declared_elements,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Persist to DB
    try:
        db_pred = Prediction(
            formula=result["formula"],
            composition_json=request.composition,
            predicted_capacity=result["specific_capacity"],
            predicted_voltage=result["voltage"],
            predicted_energy_density=result["energy_density"],
            predicted_band_gap=result["band_gap"],
            predicted_density=result["density"],
            predicted_stability=result["structural_stability"],
            predicted_ionic_conductivity=result["ionic_conductivity"],
            predicted_na_diffusion=result["na_diffusion_barrier"],
            predicted_cycle_life=result["cycle_life"],
            shap_values_json=result["shap_values"],
            explanation_text=result["explanation"],
            validity_passed=int(result["validation"]["passed"]),
            validity_details=result["validation"],
        )
        db.add(db_pred)
        db.commit()
        db.refresh(db_pred)
        result["id"] = db_pred.id
    except Exception:
        pass  # DB errors should not break prediction response

    return PredictionResponse(
        id=result.get("id"),
        formula=result["formula"],
        specific_capacity=result["specific_capacity"],
        voltage=result["voltage"],
        energy_density=result["energy_density"],
        formation_energy=result["formation_energy"],
        ionic_conductivity=result["ionic_conductivity"],
        na_diffusion_barrier=result["na_diffusion_barrier"],
        structural_stability=result["structural_stability"],
        band_gap=result["band_gap"],
        density=result["density"],
        cycle_life=result["cycle_life"],
        capacity_retention=result["capacity_retention"],
        validation=ValidationResult(**result["validation"]),
        shap_values=[SHAPEntry(**s) for s in result["shap_values"]],
        explanation=result["explanation"],
        performance_score=result["performance_score"],
    )
