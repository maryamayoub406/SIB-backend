"""
POST /generate — VAE material generation endpoint
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import GenerationRequest, GenerationResponse, GeneratedMaterialItem
from backend.models.db_models import GeneratedMaterial
from backend.ml.vae import generate_materials

router = APIRouter(prefix="/generate", tags=["Generation"])


def _save_to_db(materials: list, session_id: str, db: Session):
    """Background task: persist generated materials to DB."""
    try:
        for m in materials:
            db_mat = GeneratedMaterial(
                formula=m["formula"],
                elements=m["elements"],
                composition_json=m.get("composition_json"),
                generation_method=m.get("generation_method", "VAE"),
                performance_score=m["performance_score"],
                specific_capacity=m["specific_capacity"],
                voltage=m["voltage"],
                energy_density=m["energy_density"],
                formation_energy=m["formation_energy"],
                band_gap=m["band_gap"],
                density=m["density"],
                structural_stability=m["structural_stability"],
                charge_neutral=m.get("charge_neutral", 1),
                validity_passed=m.get("validity_passed", 1),
                rank_position=m.get("rank_position"),
                session_id=session_id,
            )
            db.add(db_mat)
        db.commit()
    except Exception as e:
        print(f"[Generate] DB save error: {e}")


@router.post("", response_model=GenerationResponse)
async def generate(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Generate novel sodium-ion battery cathode materials using VAE.
    Returns ranked list with validation and performance scores.
    """
    try:
        result = generate_materials(
            num_materials=request.num_materials,
            filter_valid_only=request.filter_valid_only,
            min_energy_density=request.min_energy_density,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    materials = result["materials"]

    # Save to DB in background
    background_tasks.add_task(_save_to_db, materials, result["session_id"], db)

    return GenerationResponse(
        session_id=result["session_id"],
        total_generated=result["total_generated"],
        valid_count=result["valid_count"],
        materials=[
            GeneratedMaterialItem(
                rank=m["rank_position"],
                formula=m["formula"],
                elements=m["elements"],
                performance_score=m["performance_score"],
                specific_capacity=m["specific_capacity"],
                voltage=m["voltage"],
                energy_density=m["energy_density"],
                formation_energy=m["formation_energy"],
                structural_stability=m["structural_stability"],
                charge_neutral=bool(m.get("charge_neutral", 1)),
                validity_passed=bool(m.get("validity_passed", 1)),
                generation_method=m.get("generation_method", "VAE"),
            )
            for m in materials
        ],
    )
