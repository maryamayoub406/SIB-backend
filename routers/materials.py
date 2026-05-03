"""
GET /material/{id}  — Retrieve a single known material by ID
GET /materials      — List materials with pagination
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List

from backend.database import get_db
from backend.schemas import MaterialDetail, SaveMaterialRequest
from backend.models.db_models import Material, SavedMaterial

router = APIRouter(prefix="/material", tags=["Materials"])


@router.post("/save")
def save_material(request: SaveMaterialRequest, db: Session = Depends(get_db)):
    """Save a generated material to the permanent library."""
    db_mat = SavedMaterial(
        formula=request.formula,
        elements=request.elements,
        performance_score=request.performance_score,
        energy_density=request.energy_density,
        voltage=request.voltage,
        specific_capacity=request.specific_capacity,
        stability=request.stability
    )
    db.add(db_mat)
    db.commit()
    db.refresh(db_mat)
    return {"status": "success", "id": db_mat.id, "formula": db_mat.formula}


@router.get("s", response_model=List[MaterialDetail])
def list_materials(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List known materials from the database."""
    materials = db.query(Material).offset(skip).limit(limit).all()
    return [MaterialDetail.model_validate(m) for m in materials]


@router.get("/{material_id}", response_model=MaterialDetail)
def get_material(material_id: int, db: Session = Depends(get_db)):
    """Retrieve a single material by its ID."""
    mat = db.query(Material).filter(Material.id == material_id).first()
    if not mat:
        raise HTTPException(status_code=404, detail=f"Material {material_id} not found")
    return MaterialDetail.model_validate(mat)
