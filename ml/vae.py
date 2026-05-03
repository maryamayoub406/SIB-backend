"""
VAE-based material generator for sodium-ion battery cathodes.
Uses composition space learned from existing training data.
"""
import os
import numpy as np
import json
import uuid
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from backend.ml.predictor import (
    predict_properties,
    validate_material,
    estimate_voltage,
    estimate_specific_capacity,
    compute_performance_score,
    estimate_ionic_conductivity,
    estimate_cycle_life,
    VALENCE,
    _PERIODIC,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "Models"
DATA_DIR = BASE_DIR / "Data" / "Training_Data"
UTILS_DIR = BASE_DIR / "Utils"

LATENT_DIM = 32
INPUT_DIM = 62
HIDDEN_DIM = 128

# ----------------------------------------------------------------
# VAE Architecture
# ----------------------------------------------------------------

class Generator(nn.Module):
    """
    Generator network: transforms random noise into material compositions
    Input: latent vector (noise)
    Output: composition vector representing element ratios
    """
    def __init__(self, latent_dim=128, output_dim=62, hidden_dims=[256, 512, 256]):
        super(Generator, self).__init__()

        layers = []
        input_dim = latent_dim

        # Hidden layers with batch normalization and LeakyReLU
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim

        # Output layer with Softmax to ensure sum = 1 (valid composition)
        layers.extend([
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=1)  # Ensures composition sums to 1
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        composition = self.model(z)
        return composition

    def generate(self, n_samples=10, temperature=1.0):
        # Using 128 dims natively for the GAN
        z = torch.randn(n_samples, 128) * temperature
        # Batch Norm requires training mode for single samples usually, but we use eval + batch
        self.eval() 
        with torch.no_grad():
            output = self.model(z)
        # Avoid relying on PyTorch's NumPy bridge at runtime; some local
        # environments have a mismatched NumPy/Torch ABI.
        return output.detach().cpu().tolist()


# ----------------------------------------------------------------
# Model singleton
# ----------------------------------------------------------------
_gan_model: Generator = None
_idx_to_elem: dict = None

def _load_vae():
    """Now loads the native GAN Model checkpoint"""
    global _gan_model, _idx_to_elem
    if _gan_model is not None:
        return _gan_model, _idx_to_elem

    with open(UTILS_DIR / "element_mappings.json") as f:
        mappings = json.load(f)
    _idx_to_elem = {int(v): k for k, v in mappings.get("element_to_idx", {}).items()}

    gan_path = BASE_DIR / "Models" / "gan_model_checkpoint.pth"
    _gan_model = Generator()

    if gan_path.exists():
        try:
            # Check structure of loaded dict
            checkpoint = torch.load(str(gan_path), map_location="cpu", weights_only=False)
            if "generator_state_dict" in checkpoint:
                _gan_model.load_state_dict(checkpoint["generator_state_dict"])
            elif "model.0.weight" in checkpoint:
                _gan_model.load_state_dict(checkpoint)
            elif "state_dict" in checkpoint:
                _gan_model.load_state_dict(checkpoint["state_dict"])
            else:
                 _gan_model.load_state_dict(checkpoint) # fallback
                 
            _gan_model.eval()
            print("[GAN] Loaded custom trained checkpoint natively.")
        except Exception as e:
            print(f"[GAN] Failed to load checkpoint ({e}). Using random intialization.")
    else:
        print("[GAN] No checkpoint found in Code folder. Using random initialization.")

    return _gan_model, _idx_to_elem


# ----------------------------------------------------------------
# Composition decoding
# ----------------------------------------------------------------
SIB_ELEMENTS = [
    "Na", "Fe", "Mn", "Co", "Ni", "V", "O", "P", "S", "F",
    "Ti", "Al", "Cr", "Nb", "Zr", "Cu", "Mg", "Ca",
]

ELEMENT_FORMULA_WEIGHTS = {
    "Na": 1, "O": 2, "P": 1, "S": 1, "F": 1,
    "Fe": 1, "Mn": 1, "Co": 1, "Ni": 1, "V": 1,
    "Ti": 1, "Al": 1, "Cr": 1, "Nb": 1,
}


def _vector_to_composition(vec: np.ndarray, idx_to_elem: dict, threshold=0.03) -> Tuple[dict, List[str]]:
    """Convert probability vector to composition dict and list of declared elements."""
    comp = {}
    declared = []
    
    for i, v in enumerate(vec):
        if v > (threshold / 2) and i in idx_to_elem:
            elem = idx_to_elem[i]
            declared.append(elem)
            if v >= threshold:  # Only add to composition if above strict threshold
                comp[elem] = float(v)

    # Ensure Na is present
    if "Na" not in comp or comp.get("Na", 0) < 0.05:
        if "Na" in {v for v in idx_to_elem.values()}:
            na_idx = next(k for k, v in idx_to_elem.items() if v == "Na")
            if na_idx < len(vec):
                comp["Na"] = 0.15
                if "Na" not in declared:
                    declared.append("Na")

    # Normalize
    total = sum(comp.values())
    if total > 0:
        comp = {k: v / total for k, v in comp.items()}

    return comp, declared


def _composition_to_formula(comp: dict) -> str:
    """Convert fractional composition to a human-readable formula."""
    # Scale to reasonable integer-like stoichiometry
    if not comp:
        return "Unknown"

    # Sort by electronegativity convention (metals first, then O, then anions)
    order = {"Na": 0, "Li": 1, "K": 2, "Mg": 3, "Ca": 4,
             "Fe": 5, "Mn": 5, "Co": 5, "Ni": 5, "V": 5,
             "Ti": 5, "Cr": 5, "Al": 5, "Nb": 5, "Zr": 5,
             "P": 6, "S": 7, "O": 8, "F": 9}

    sorted_elems = sorted(comp.keys(), key=lambda e: order.get(e, 10))
    parts = []
    
    # Scale fractions explicitly to a base to ensure variations aren't lost
    for e in sorted_elems:
        f = comp[e]
        # Allow quarter-step stoich variants instead of trapping inside generic halves
        stoich = round(f * 12) / 4  
        if stoich <= 0:
            continue
        if stoich == 1.0:
            parts.append(e)
        elif stoich == int(stoich):
            parts.append(f"{e}{int(stoich)}")
        else:
            parts.append(f"{e}{stoich:.1f}")
    return "".join(parts) if parts else "Na2O"


def generate_materials(
    num_materials: int = 10,
    filter_valid_only: bool = True,
    min_energy_density: float = 200.0,
    temperature: float = 1.0,
) -> dict:
    """
    Main generation function.
    Returns list of generated materials with predicted properties.
    """
    vae, idx_to_elem = _load_vae()
    session_id = str(uuid.uuid4())[:8]

    materials = []
    seen_formulas = set()
    max_attempts = 5
    attempts = 0

    while len(materials) < num_materials and attempts < max_attempts:
        attempts += 1
        # Generate a batch proportional to the remaining count, but at least 100
        batch_size = max((num_materials - len(materials)) * 20, 100)
        raw_vectors = vae.generate(batch_size, temperature=temperature)

        for vec in raw_vectors:
            comp, declared = _vector_to_composition(np.array(vec), idx_to_elem)
            if not comp:
                continue

            formula = _composition_to_formula(comp)
            if formula in seen_formulas:
                continue
            seen_formulas.add(formula)

            try:
                props = predict_properties(formula, declared_elements=declared)
            except Exception:
                continue

            validity = props["validation"]

            if filter_valid_only and not validity.get("passed", False):
                continue

            if props["energy_density"] < min_energy_density:
                continue

            material = {
                "formula": formula,
                "elements": " ".join(sorted(comp.keys())),
                "composition_json": comp,
                "generation_method": "VAE",
                "performance_score": props["performance_score"],
                "specific_capacity": props["specific_capacity"],
                "voltage": props["voltage"],
                "energy_density": props["energy_density"],
                "formation_energy": props["formation_energy"],
                "band_gap": props["band_gap"],
                "density": props["density"],
                "structural_stability": props["structural_stability"],
                "charge_neutral": int(validity.get("charge_neutral", False)),
                "validity_passed": int(validity.get("passed", False)),
                "session_id": session_id,
            }
            materials.append(material)

            if len(materials) >= num_materials:
                break

    # Sort by performance score
    materials.sort(key=lambda x: x["performance_score"], reverse=True)
    for i, m in enumerate(materials):
        m["rank_position"] = i + 1

    return {
        "session_id": session_id,
        "total_generated": attempts * 100, # Approximate total across attempts
        "valid_count": len(materials),
        "materials": materials
    }
