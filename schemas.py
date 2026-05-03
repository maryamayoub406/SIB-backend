"""
Pydantic schemas for request/response validation.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ----------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------
class PredictionRequest(BaseModel):
    formula: str = Field(...)
    composition: Optional[Dict[str, float]] = Field(None)
    declared_elements: Optional[List[str]] = Field(None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "formula": "NaFeO2",
                "composition": {"Na": 0.25, "Fe": 0.25, "O": 0.5}
            }
        }
    }


class SHAPEntry(BaseModel):
    element: str
    shap_value: float
    contribution: str  # "positive" | "negative"


class ValidationResult(BaseModel):
    formation_energy_ok: bool
    charge_neutral: bool
    chemical_validity: bool
    structural_feasibility: bool
    element_consistent: bool
    passed: bool
    # Extended validation detail (optional for backwards compat)
    redox_active: Optional[bool] = None
    structural_plausibility: Optional[bool] = None
    charge_imbalance: Optional[float] = None
    phantom_elements: Optional[List[str]] = Field(default_factory=list)
    missing_elements: Optional[List[str]] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    id: Optional[int] = None
    formula: str
    # Core properties
    specific_capacity: float = Field(..., description="mAh/g")
    voltage: float = Field(..., description="V")
    energy_density: float = Field(..., description="Wh/kg")
    formation_energy: float = Field(..., description="eV/atom")
    ionic_conductivity: float = Field(..., description="S/cm (log scale)")
    na_diffusion_barrier: float = Field(..., description="eV")
    structural_stability: float = Field(..., description="eV/atom")
    band_gap: float = Field(..., description="eV")
    density: float = Field(..., description="g/cm3")
    cycle_life: int = Field(..., description="estimated cycles")
    capacity_retention: float = Field(..., description="0-1 fraction")
    # Validation
    validation: ValidationResult
    # Explainability
    shap_values: List[SHAPEntry]
    explanation: str
    # Score
    performance_score: float = Field(..., description="0-100 composite score")
    created_at: Optional[datetime] = None


# ----------------------------------------------------------------
# GENERATION
# ----------------------------------------------------------------
class GenerationRequest(BaseModel):
    num_materials: int = Field(10, ge=1, le=100)
    filter_valid_only: bool = True
    min_energy_density: float = Field(200.0, description="Wh/kg filter")
    temperature: float = Field(1.0, description="VAE sampling temperature")

    model_config = {
        "json_schema_extra": {
            "example": {
                "num_materials": 10,
                "filter_valid_only": True,
                "min_energy_density": 200.0,
                "temperature": 1.0
            }
        }
    }


class GeneratedMaterialItem(BaseModel):
    rank: int
    formula: str
    elements: str
    performance_score: float
    specific_capacity: float
    voltage: float
    energy_density: float
    formation_energy: float
    structural_stability: float
    charge_neutral: bool
    validity_passed: bool
    generation_method: str


class GenerationResponse(BaseModel):
    session_id: str
    total_generated: int
    valid_count: int
    materials: List[GeneratedMaterialItem]


# ----------------------------------------------------------------
# DEGRADATION
# ----------------------------------------------------------------
class DegradationRequest(BaseModel):
    formula: str = Field(...)
    initial_capacity: float = Field(150.0, description="mAh/g")
    total_cycles: int = Field(500, ge=10, le=5000)
    temperature_c: float = Field(25.0, description="Operating temperature °C")
    c_rate: float = Field(1.0, description="C-rate (1C, 2C, etc.)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "formula": "NaFeO2",
                "initial_capacity": 150.0,
                "total_cycles": 500,
                "temperature_c": 25.0,
                "c_rate": 1.0
            }
        }
    }


class CyclePoint(BaseModel):
    cycle: int
    capacity: float
    voltage: float
    degradation_index: float


class DegradationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: Optional[int] = None
    formula: str
    total_cycles: int
    initial_capacity: float
    final_capacity: float
    capacity_retention: float
    voltage_decay: float
    cycle_life_predicted: int
    degradation_index: float
    curve_data: List[CyclePoint]
    model_type: str
    summary: str


# ----------------------------------------------------------------
# RANKING
# ----------------------------------------------------------------
class RankRequest(BaseModel):
    formulas: Optional[List[str]] = None
    top_n: int = Field(20, ge=1, le=100)
    sort_by: str = Field("energy_density", description="Sort field")


class RankedMaterial(BaseModel):
    rank: int
    formula: str
    energy_density: float
    voltage: float
    specific_capacity: float
    stability: float
    performance_score: float


class RankResponse(BaseModel):
    total: int
    ranked: List[RankedMaterial]


# ----------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------
class DashboardStats(BaseModel):
    total_materials_in_db: int
    total_predictions: int
    total_generated: int
    total_degradation_runs: int
    top_energy_density: float
    avg_energy_density: float
    top_formula: str
    system_status: str
    model_accuracy: float = 52.7

    model_config = {"protected_namespaces": ()}


# ----------------------------------------------------------------
# MATERIAL DETAIL
# ----------------------------------------------------------------
class MaterialDetail(BaseModel):
    id: int
    formula: str
    elements: str
    num_elements: int
    specific_capacity: Optional[float]
    voltage: Optional[float]
    energy_density: Optional[float]
    formation_energy: Optional[float]
    band_gap: Optional[float]
    density: Optional[float]
    structural_stability: Optional[float]
    cycle_life: Optional[int]
    source: str
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ----------------------------------------------------------------
# SAVED MATERIALS
# ----------------------------------------------------------------
class SaveMaterialRequest(BaseModel):
    formula: str
    elements: str
    performance_score: float
    energy_density: float
    voltage: float
    specific_capacity: float
    stability: float
