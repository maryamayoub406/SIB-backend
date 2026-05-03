"""
SQLAlchemy ORM models matching database/schema.sql
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text,
    JSON, Enum, ForeignKey, SmallInteger, Index
)
from sqlalchemy.orm import relationship
from backend.database import Base
import enum


class UserRole(str, enum.Enum):
    admin = "admin"
    researcher = "researcher"
    viewer = "viewer"


class GenerationMethod(str, enum.Enum):
    GAN = "GAN"
    VAE = "VAE"
    diffusion = "diffusion"


# ----------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum("admin", "researcher", "viewer"), default="viewer")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")


# ----------------------------------------------------------------
class Material(Base):
    __tablename__ = "materials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    formula = Column(String(100), nullable=False, index=True)
    elements = Column(String(255))
    num_elements = Column(Integer)
    specific_capacity = Column(Float, comment="mAh/g")
    voltage = Column(Float, comment="V")
    energy_density = Column(Float, comment="Wh/kg")
    formation_energy = Column(Float, comment="eV/atom")
    ionic_conductivity = Column(Float, comment="S/cm")
    na_diffusion_barrier = Column(Float, comment="eV")
    structural_stability = Column(Float, comment="eV/atom hull distance")
    band_gap = Column(Float, comment="eV")
    density = Column(Float, comment="g/cm3")
    cycle_life = Column(Integer)
    capacity_retention = Column(Float)
    charge_neutral = Column(SmallInteger, default=1)
    source = Column(String(100), default="materials_project")
    created_at = Column(DateTime, default=datetime.utcnow)

    degradation_results = relationship("DegradationResult", back_populates="material")


# ----------------------------------------------------------------
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    formula = Column(String(100), nullable=False, index=True)
    composition_json = Column(JSON)
    predicted_capacity = Column(Float)
    predicted_voltage = Column(Float)
    predicted_energy_density = Column(Float)
    predicted_band_gap = Column(Float)
    predicted_density = Column(Float)
    predicted_stability = Column(Float)
    predicted_ionic_conductivity = Column(Float)
    predicted_na_diffusion = Column(Float)
    predicted_cycle_life = Column(Integer)
    shap_values_json = Column(JSON)
    explanation_text = Column(Text)
    validity_passed = Column(SmallInteger, default=1)
    validity_details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))

    user = relationship("User", back_populates="predictions")


# ----------------------------------------------------------------
class GeneratedMaterial(Base):
    __tablename__ = "generated_materials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    formula = Column(String(100), index=True)
    elements = Column(String(255))
    composition_json = Column(JSON)
    generation_method = Column(
        Enum("GAN", "VAE", "diffusion"), default="VAE"
    )
    performance_score = Column(Float, index=True)
    specific_capacity = Column(Float)
    voltage = Column(Float)
    energy_density = Column(Float)
    formation_energy = Column(Float)
    band_gap = Column(Float)
    density = Column(Float)
    structural_stability = Column(Float)
    charge_neutral = Column(SmallInteger)
    validity_passed = Column(SmallInteger, default=1)
    rank_position = Column(Integer)
    session_id = Column(String(100), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ----------------------------------------------------------------
class DegradationResult(Base):
    __tablename__ = "degradation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, ForeignKey("materials.id", ondelete="SET NULL"))
    formula = Column(String(255))
    total_cycles = Column(Integer)
    initial_capacity = Column(Float)
    final_capacity = Column(Float)
    capacity_retention = Column(Float)
    voltage_decay = Column(Float)
    cycle_life_predicted = Column(Integer)
    degradation_index = Column(Float)
    curve_data_json = Column(JSON)
    model_type = Column(String(50), default="LSTM")
    created_at = Column(DateTime, default=datetime.utcnow)

    material = relationship("Material", back_populates="degradation_results")


# ----------------------------------------------------------------
class SavedMaterial(Base):
    __tablename__ = "saved_materials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    formula = Column(String(100), index=True)
    elements = Column(String(255))
    performance_score = Column(Float)
    energy_density = Column(Float)
    voltage = Column(Float)
    specific_capacity = Column(Float)
    stability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
