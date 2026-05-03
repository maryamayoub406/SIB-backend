"""
ML Property Predictor
Wraps existing trained models (model_bandgap.pkl, model_density.pkl, model_stability.pkl)
and adds extended property computation + SHAP explainability.

FIXES APPLIED (v2):
  1. Charge neutrality threshold tightened: 0.5 → 0.05
  2. Invalid material score penalty hardened: max 45 → max 20
  3. Added validate_element_consistency() for phantom-element detection
  4. estimate_voltage() now uses raw stoichiometric amounts (not normalized fractions)
  5. Removed artificial energy-density floor (50 Wh/kg) for invalid materials
  6. Removed forced 50% utilization floor in estimate_specific_capacity()
  7. Improved check_charge_balance() — iterative product approach, no recursion risk
  8. formation_energy fallback changed from -1.5 to -0.5 to avoid artificial "excellent stability"
"""

import os
import json
import hashlib
import math
import re
import numpy as np
import joblib
from pathlib import Path
from itertools import product as itertools_product
import importlib

shap = None

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # FYP/
load_dotenv(BASE_DIR / "backend" / ".env")

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "Models")).resolve()
UTILS_DIR = Path(os.getenv("UTILS_DIR", BASE_DIR / "Utils")).resolve()
PROCESSED_DATA_PATH = Path(
    os.getenv("PROCESSED_TRAINING_DATA_PATH", BASE_DIR / "Data" / "Processed" / "mp_from_scratch_training.pkl")
).resolve()


def _load_models():
    """Load all three sklearn models and element mappings."""
    model_bg = joblib.load(MODEL_DIR / "model_bandgap.pkl")
    model_dens = joblib.load(MODEL_DIR / "model_density.pkl")
    model_stab = joblib.load(MODEL_DIR / "new_model_stability.pkl")
    _set_runtime_n_jobs(model_bg)
    _set_runtime_n_jobs(model_dens)
    _set_runtime_n_jobs(model_stab)
    with open(UTILS_DIR / "element_mappings.json") as f:
        mappings = json.load(f)
    return model_bg, model_dens, model_stab, mappings


# Module-level singletons (lazy load on first use)
_model_bg = None
_model_dens = None
_model_stab = None
_mappings = None
_explainer_bg = None
_descriptor_lookup = None
_family_descriptor_lookup = None
_transition_categories = None


def _set_runtime_n_jobs(model, value: int = 1):
    """Force loaded sklearn models to use a safe local worker count."""
    seen = set()
    stack = [model]

    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        if hasattr(current, "n_jobs"):
            try:
                current.set_params(n_jobs=value)
            except Exception:
                try:
                    setattr(current, "n_jobs", value)
                except Exception:
                    pass

        for attr in ("estimators", "estimators_"):
            if not hasattr(current, attr):
                continue
            estimators = getattr(current, attr, None)
            if not estimators:
                continue
            for item in estimators:
                stack.append(item[1] if isinstance(item, tuple) else item)

        for attr in ("final_estimator", "final_estimator_"):
            if hasattr(current, attr):
                stack.append(getattr(current, attr))


def get_models():
    global _model_bg, _model_dens, _model_stab, _mappings
    if _model_bg is None:
        _model_bg, _model_dens, _model_stab, _mappings = _load_models()
    return _model_bg, _model_dens, _model_stab, _mappings


def _amounts_from_formula(formula: str):
    try:
        from pymatgen.core import Composition

        comp = Composition(formula)
        data = {str(k): float(v) for k, v in comp.get_el_amt_dict().items()}
        total = float(sum(data.values()))
        if total > 0:
            return data, total
    except Exception:
        pass

    # Use raw (un-normalized) parser
    raw = _parse_formula_raw(formula)
    if raw:
        return {k: float(v) for k, v in raw.items()}, float(sum(raw.values()))
    return {}, 0.0


def _parse_formula_raw(formula: str) -> dict:
    """Parse formula returning raw stoichiometric amounts (NOT normalized)."""
    pattern = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)')
    matches = pattern.findall(formula)
    comp = {}
    for elem, cnt in matches:
        if elem not in _PERIODIC:
            continue
        c = float(cnt) if cnt else 1.0
        comp[elem] = comp.get(elem, 0.0) + c
    return comp


def _family_name_from_amounts(amounts: dict) -> str:
    if amounts.get("P", 0.0) > 0:
        return "phosphate"
    if amounts.get("F", 0.0) > 0:
        return "fluoride"
    if amounts.get("S", 0.0) > 0:
        return "sulfate"
    if amounts.get("Si", 0.0) > 0:
        return "silicate"
    return "oxide"


def _infer_transition_element(amounts: dict) -> str:
    tm_set = {"Mn", "Fe", "Co", "Ni", "V", "Ti", "Cr", "Nb", "Zr", "Cu"}
    best_elem = ""
    best_amt = -1.0
    for elem in tm_set:
        amt = amounts.get(elem, 0.0)
        if amt > best_amt:
            best_amt = amt
            best_elem = elem
    return best_elem if best_amt > 0 else "UNK"


def _build_family_key(formula: str) -> str:
    amounts, _ = _amounts_from_formula(formula)
    return f"{_infer_transition_element(amounts)}::{_family_name_from_amounts(amounts)}"


def _load_descriptor_tables():
    global _descriptor_lookup, _family_descriptor_lookup, _transition_categories
    if _descriptor_lookup is not None:
        return _descriptor_lookup, _family_descriptor_lookup, _transition_categories

    if not PROCESSED_DATA_PATH.exists():
        _descriptor_lookup = {}
        _family_descriptor_lookup = {}
        _transition_categories = []
        return _descriptor_lookup, _family_descriptor_lookup, _transition_categories

    try:
        import pandas as pd

        df = pd.read_pickle(PROCESSED_DATA_PATH)
    except Exception:
        _descriptor_lookup = {}
        _family_descriptor_lookup = {}
        _transition_categories = []
        return _descriptor_lookup, _family_descriptor_lookup, _transition_categories

    descriptor_cols = [
        "avg_electronegativity",
        "avg_atomic_mass",
        "avg_atomic_radius",
        "num_elements",
        "total_atoms",
        "formation_energy_per_atom",
        "nsites",
        "volume",
        "nelements",
        "theoretical",
        "volume_per_site",
        "duplicate_count",
        "band_gap_std",
        "density_std",
        "energy_above_hull_std",
        "quality_score",
        "queried_transition_element",
    ]
    missing = [col for col in descriptor_cols if col not in df.columns]
    for col in missing:
        df[col] = 0.0 if col != "queried_transition_element" else ""

    _transition_categories = sorted(
        {str(v).strip() for v in df["queried_transition_element"].fillna("") if str(v).strip()}
    )
    df = df.copy()
    df["family_key"] = df["formula"].astype(str).map(_build_family_key)

    family_cols = [c for c in descriptor_cols if c != "queried_transition_element"]
    family_medians = (
        df.groupby("family_key")[family_cols]
        .median(numeric_only=True)
        .to_dict(orient="index")
    )

    descriptor_lookup = {}
    for _, row in df.iterrows():
        descriptor_lookup[str(row["formula"]).strip()] = {col: row[col] for col in descriptor_cols}

    _descriptor_lookup = descriptor_lookup
    _family_descriptor_lookup = family_medians
    return _descriptor_lookup, _family_descriptor_lookup, _transition_categories


def _fallback_descriptor_values(formula: str, composition: dict) -> dict:
    amounts, total = _amounts_from_formula(formula)
    total = total or 1.0

    atomic_mass = {
        "Na": 22.99, "Fe": 55.85, "Mn": 54.94, "Co": 58.93, "Ni": 58.69, "V": 50.94, "O": 16.0,
        "P": 30.97, "S": 32.06, "F": 19.0, "Ti": 47.87, "Al": 26.98, "Cr": 52.0, "Zr": 91.22,
        "Nb": 92.91, "Cu": 63.55, "Si": 28.09,
    }
    atomic_radius = {
        "Na": 1.86, "Fe": 1.56, "Mn": 1.61, "Co": 1.52, "Ni": 1.49, "V": 1.71, "O": 0.66,
        "P": 1.07, "S": 1.05, "F": 0.57, "Ti": 1.76, "Cr": 1.66, "Zr": 2.06, "Nb": 1.98,
        "Cu": 1.45, "Si": 1.11,
    }
    weights = {k: v / total for k, v in amounts.items()} if amounts else composition
    elems = list(weights.keys())
    fracs = list(weights.values())
    en_vals = [ELECTRONEGATIVITY.get(e, 2.0) for e in elems]
    mean_en = float(np.average(en_vals, weights=fracs)) if fracs else 2.0
    avg_mass = float(np.average([atomic_mass.get(e, 40.0) for e in elems], weights=fracs)) if fracs else 0.0
    avg_radius = float(np.average([atomic_radius.get(e, 1.2) for e in elems], weights=fracs)) if fracs else 0.0
    total_atoms = float(total)
    nsites = max(total_atoms * 4.0, 4.0)
    volume_per_site = avg_radius * 10.0 + avg_mass * 0.02
    volume = volume_per_site * nsites
    return {
        "avg_electronegativity": mean_en,
        "avg_atomic_mass": avg_mass,
        "avg_atomic_radius": avg_radius,
        "num_elements": float(len(elems)),
        "total_atoms": total_atoms,
        # FIX 8: Changed from -1.5 to -0.5 — neutral/borderline fallback,
        # prevents ALL unknown materials getting artificially "excellent" stability
        "formation_energy_per_atom": -0.5,
        "nsites": nsites,
        "volume": volume,
        "nelements": float(len(elems)),
        "theoretical": 1.0,
        "volume_per_site": volume_per_site,
        "duplicate_count": 1.0,
        "band_gap_std": 0.0,
        "density_std": 0.0,
        "energy_above_hull_std": 0.0,
        "quality_score": 1.0,
        "queried_transition_element": _infer_transition_element(amounts),
    }


def get_descriptor_values(formula: str, composition: dict) -> tuple[dict, list[str]]:
    descriptor_lookup, family_lookup, transition_categories = _load_descriptor_tables()
    formula_key = str(formula).strip()
    source = []

    if formula_key in descriptor_lookup:
        source.append("formula_lookup")
        values = descriptor_lookup[formula_key].copy()
    else:
        values = _fallback_descriptor_values(formula, composition)
        family_key = _build_family_key(formula)
        if family_key in family_lookup:
            source.append("family_median")
            for key, value in family_lookup[family_key].items():
                values[key] = float(value)
        else:
            source.append("heuristic")

    tm_name = str(values.get("queried_transition_element", "")).strip()
    if not tm_name:
        values["queried_transition_element"] = _infer_transition_element(_amounts_from_formula(formula)[0])
    return values, transition_categories


def get_shap_explainer():
    global _explainer_bg, shap
    if os.getenv("ENABLE_SHAP", "0") != "1":
        raise RuntimeError("SHAP is disabled in the current runtime")
    if shap is None:
        try:
            shap = importlib.import_module("shap")
        except Exception as exc:
            raise RuntimeError("SHAP is unavailable in the current runtime") from exc
    if _explainer_bg is None:
        model_bg, _, _, _ = get_models()
        _explainer_bg = shap.TreeExplainer(model_bg)
    return _explainer_bg


# ----------------------------------------------------------------
# Utility: composition string → element dict
# ----------------------------------------------------------------
_PERIODIC = [
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]

ELECTRONEGATIVITY = {
    "Na": 0.93, "Fe": 1.83, "Mn": 1.55, "Co": 1.88, "Ni": 1.91,
    "V": 1.63, "O": 3.44, "P": 2.19, "S": 2.58, "F": 3.98,
    "Ti": 1.54, "Al": 1.61, "Zr": 1.33, "Nb": 1.6, "Cr": 1.66,
    "Cu": 1.90, "Li": 0.98, "K": 0.82, "Ca": 1.00,
}

# FIX: V default changed from 5 → 4 (most common in Na-ion oxide cathodes)
# Mg, Li, Cu, Zn added to complete the set
VALENCE = {
    "Na": 1,  "Fe": 3,  "Mn": 4,  "Co": 3,  "Ni": 2,
    "V":  4,  "O": -2,  "P":  5,  "S":  6,  "F": -1,
    "Ti": 4,  "Al": 3,  "Zr": 4,  "Nb": 5,  "Cr": 3,
    "Mg": 2,  "Li": 1,  "Cu": 2,  "Zn": 2,
}


def parse_formula(formula: str) -> dict:
    """
    Formula parser: extracts {element: normalized_fraction}.
    Handles NaFeO2, Na2MnO3, NaFe0.5Mn0.5O2 etc.
    NOTE: Returns NORMALIZED fractions (sum=1).
    Use _parse_formula_raw() or _amounts_from_formula() for raw stoichiometric amounts.
    """
    pattern = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)')
    matches = pattern.findall(formula)
    comp = {}
    total = 0.0
    for elem, cnt in matches:
        if elem not in _PERIODIC:
            continue
        c = float(cnt) if cnt else 1.0
        comp[elem] = comp.get(elem, 0.0) + c
        total += c
    if total > 0:
        comp = {k: v / total for k, v in comp.items()}
    return comp


def formula_hash_features(formula: str, n_features: int = 16) -> np.ndarray:
    """Deterministic formula-level features to help separate repeated compositions."""
    digest = hashlib.sha256(formula.encode("utf-8")).digest()
    return np.array([digest[i] / 255.0 for i in range(n_features)], dtype=float)


def composition_to_vector(composition: dict, mappings: dict, formula: str = "") -> np.ndarray:
    """Convert element:fraction dict to model feature vector."""
    vec = np.zeros(62)
    for elem, frac in composition.items():
        if elem in mappings.get("element_to_idx", {}):
            idx = mappings["element_to_idx"][elem]
            if idx < 62:
                vec[idx] = frac

    elements = list(composition.keys())
    fractions = list(composition.values())

    en_vals = [ELECTRONEGATIVITY.get(e, 2.0) for e in elements]
    mean_en = np.average(en_vals, weights=fractions) if fractions else 2.0
    var_en = np.average([(e - mean_en) ** 2 for e in en_vals], weights=fractions) if fractions else 0.0
    n_elem = float(len(elements))
    na_frac = composition.get("Na", 0.0)
    o_frac = composition.get("O", 0.0)
    TM = {"Fe", "Mn", "Co", "Ni", "V", "Ti", "Cr", "Nb", "Zr", "Cu"}
    tm_frac = sum(composition.get(e, 0.0) for e in TM)
    val_sum = sum(VALENCE.get(e, 0) * f for e, f in composition.items())
    val_balance = abs(val_sum)
    p_frac = composition.get("P", 0.0)
    f_frac = composition.get("F", 0.0)
    s_frac = composition.get("S", 0.0)

    chem_features = np.array([
        mean_en, var_en, n_elem, na_frac, o_frac,
        tm_frac, val_balance, p_frac, f_frac, s_frac
    ])

    if formula:
        return np.concatenate([vec, chem_features, formula_hash_features(formula)])
    return np.concatenate([vec, chem_features, np.zeros(16)])


def build_domain_features(formula: str, descriptor_values: dict) -> np.ndarray:
    amounts, total = _amounts_from_formula(formula)
    total = total or 1.0

    tm_set = {"Mn", "Fe", "Co", "Ni", "V", "Ti", "Cr", "Nb", "Zr", "Cu"}
    alkali_set = {"Na", "Li", "K"}
    anion_set = {"O", "F", "S", "P", "Si", "Cl"}

    tm_total = sum(amounts.get(e, 0.0) for e in tm_set)
    na_total = amounts.get("Na", 0.0)
    o_total = amounts.get("O", 0.0)
    p_total = amounts.get("P", 0.0)
    f_total = amounts.get("F", 0.0)
    s_total = amounts.get("S", 0.0)
    si_total = amounts.get("Si", 0.0)

    n_tm_species = sum(1 for e in tm_set if amounts.get(e, 0.0) > 0)
    n_alkali_species = sum(1 for e in alkali_set if amounts.get(e, 0.0) > 0)
    n_anion_species = sum(1 for e in anion_set if amounts.get(e, 0.0) > 0)

    na_to_tm = na_total / tm_total if tm_total else 0.0
    o_to_tm = o_total / tm_total if tm_total else 0.0
    anion_to_tm = sum(amounts.get(e, 0.0) for e in anion_set) / tm_total if tm_total else 0.0

    layered_oxide_flag = float(o_total > 0 and tm_total > 0 and p_total == 0 and s_total == 0 and f_total == 0 and si_total == 0)
    phosphate_flag = float(p_total > 0)
    sulfate_flag = float(s_total > 0)
    fluoride_flag = float(f_total > 0)
    silicate_flag = float(si_total > 0)
    mixed_tm_flag = float(n_tm_species > 1)

    formation_energy = float(descriptor_values.get("formation_energy_per_atom", -0.5) or -0.5)
    nsites = float(descriptor_values.get("nsites", 4.0) or 4.0)
    volume_per_site = float(descriptor_values.get("volume_per_site", 10.0) or 10.0)
    nelements = float(descriptor_values.get("nelements", len(amounts)) or len(amounts))
    theoretical = float(descriptor_values.get("theoretical", 1.0) or 1.0)
    raw_elements = len(re.findall(r"[A-Z][a-z]?", formula))

    return np.array(
        [
            tm_total / total,
            na_total / total,
            o_total / total,
            p_total / total,
            f_total / total,
            s_total / total,
            si_total / total,
            float(n_tm_species),
            float(n_alkali_species),
            float(n_anion_species),
            na_to_tm,
            o_to_tm,
            anion_to_tm,
            layered_oxide_flag,
            phosphate_flag,
            sulfate_flag,
            fluoride_flag,
            silicate_flag,
            mixed_tm_flag,
            abs(formation_energy),
            math.log1p(max(nsites, 0.0)),
            math.log1p(max(volume_per_site, 0.0)),
            nelements,
            float(raw_elements),
            theoretical,
        ],
        dtype=float,
    )


def build_runtime_feature_vector(formula: str, composition: dict, mappings: dict) -> np.ndarray:
    base_features = composition_to_vector(composition, mappings, formula)
    descriptor_values, transition_categories = get_descriptor_values(formula, composition)
    extra_numeric = np.array(
        [
            float(descriptor_values.get("avg_electronegativity", 0.0) or 0.0),
            float(descriptor_values.get("avg_atomic_mass", 0.0) or 0.0),
            float(descriptor_values.get("avg_atomic_radius", 0.0) or 0.0),
            float(descriptor_values.get("num_elements", 0.0) or 0.0),
            float(descriptor_values.get("total_atoms", 0.0) or 0.0),
            float(descriptor_values.get("formation_energy_per_atom", 0.0) or 0.0),
            float(descriptor_values.get("nsites", 0.0) or 0.0),
            float(descriptor_values.get("volume", 0.0) or 0.0),
            float(descriptor_values.get("nelements", 0.0) or 0.0),
            float(descriptor_values.get("theoretical", 0.0) or 0.0),
            float(descriptor_values.get("volume_per_site", 0.0) or 0.0),
        ],
        dtype=float,
    )
    domain_features = build_domain_features(formula, descriptor_values)
    transition_vec = np.zeros(len(transition_categories), dtype=float)
    tm_name = str(descriptor_values.get("queried_transition_element", "")).strip()
    if tm_name and tm_name in transition_categories:
        transition_vec[transition_categories.index(tm_name)] = 1.0
    return np.concatenate([base_features, extra_numeric, domain_features, transition_vec])


def build_stability_feature_vector_v2(formula: str, descriptor_values: dict) -> np.ndarray:
    """Build the 22-feature vector for the new XGBoost model."""
    amounts, total = _amounts_from_formula(formula)
    total = total or 1.0

    tm_list = ["Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr", "Cu", "Zn", "Nb", "Zr"]
    props = ["X", "r", "val", "mass", "period", "group"]
    element_props = {
        "Na": {"X": 0.93, "r": 1.86, "val": 1, "mass": 22.99, "period": 3, "group": 1},
        "Fe": {"X": 1.83, "r": 1.26, "val": 8, "mass": 55.85, "period": 4, "group": 8},
        "Mn": {"X": 1.55, "r": 1.29, "val": 7, "mass": 54.94, "period": 4, "group": 7},
        "Co": {"X": 1.88, "r": 1.25, "val": 9, "mass": 58.93, "period": 4, "group": 9},
        "Ni": {"X": 1.91, "r": 1.24, "val": 10, "mass": 58.69, "period": 4, "group": 10},
        "O":  {"X": 3.44, "r": 0.73, "val": 6, "mass": 16.00, "period": 2, "group": 16},
        "P":  {"X": 2.19, "r": 1.07, "val": 5, "mass": 30.97, "period": 3, "group": 15},
        "S":  {"X": 2.58, "r": 1.02, "val": 6, "mass": 32.06, "period": 3, "group": 16},
        "F":  {"X": 3.98, "r": 0.64, "val": 7, "mass": 19.00, "period": 2, "group": 17},
        "Ti": {"X": 1.54, "r": 1.47, "val": 4, "mass": 47.87, "period": 4, "group": 4},
        "V":  {"X": 1.63, "r": 1.35, "val": 5, "mass": 50.94, "period": 4, "group": 5},
        "Cr": {"X": 1.66, "r": 1.29, "val": 6, "mass": 52.00, "period": 4, "group": 6},
    }
    default_props = {"X": 2.0, "r": 1.3, "val": 4, "mass": 40.0, "period": 4, "group": 6}

    weighted = {prop: 0.0 for prop in props}
    prop_values = {prop: [] for prop in props}
    for elem, amt in amounts.items():
        info = element_props.get(elem, default_props)
        frac = amt / total
        for prop in props:
            weighted[prop] += frac * info[prop]
            prop_values[prop].append(info[prop])

    diffs = {
        prop: (max(prop_values[prop]) - min(prop_values[prop])) if len(prop_values[prop]) > 1 else 0.0
        for prop in props
    }

    return np.array([
        float(len(amounts)),
        float(total),
        amounts.get("Na", 0.0) / total,
        amounts.get("O", 0.0) / total,
        float(any(elem in amounts for elem in tm_list)),
        sum(amounts.get(elem, 0.0) for elem in tm_list) / total,
        weighted["X"],
        diffs["X"],
        weighted["r"],
        diffs["r"],
        weighted["val"],
        diffs["val"],
        weighted["mass"],
        diffs["mass"],
        weighted["period"],
        weighted["group"],
        float(descriptor_values.get("volume", 100.0) or 100.0),
        float(descriptor_values.get("nsites", 4.0) or 4.0),
        float(descriptor_values.get("nelements", len(amounts)) or len(amounts)),
        1.0,  # spacegroup_number (fallback)
        1.0,  # is_stable (fallback)
        0.0,  # energy_above_hull (fallback)
    ], dtype=float)


# ----------------------------------------------------------------
# Derived property estimators (SIB domain knowledge)
# ----------------------------------------------------------------

def estimate_voltage(composition: dict, band_gap: float, formula: str = "") -> float:
    """
    Estimate operating voltage based on composition and band gap.
    FIX 4: Uses raw stoichiometric amounts (via formula) instead of
    normalized fractions so element-ratio adjustments are accurate.
    """
    if formula:
        amounts, _ = _amounts_from_formula(formula)
    else:
        amounts = composition  # Fallback to whatever is passed

    base = 3.5
    if "P" in amounts and "V" in amounts:
        # NASICON-type
        base = 3.7
    elif "O" in amounts and "Mn" in amounts:
        # Layered Mn oxide — use raw Mn amount, capped
        mn_amt = amounts.get("Mn", 0.0)
        base = 3.2 + min(mn_amt * 0.15, 0.35)
    elif "O" in amounts and "Fe" in amounts:
        base = 3.1
    elif "O" in amounts and "Co" in amounts:
        base = 3.4
    elif "F" in amounts:
        base = 3.9

    base -= band_gap * 0.15
    return float(np.clip(base, 2.0, 4.5))


def estimate_specific_capacity(composition: dict, density: float, formula: str = "") -> float:
    """
    Estimate specific capacity (mAh/g) using raw stoichiometric amounts.
    
    FIX: Replaced normalized-fraction utilization with Faraday's Law applied
    to raw formula unit amounts. Previous version used min(na_frac * 4, 1.0)
    which decoupled the numerator and denominator, causing 3x-6x overestimation.
    """
    ATOMIC_MASS = {
        "Na": 22.99, "Fe": 55.85, "Mn": 54.94, "Co": 58.93,
        "Ni": 58.69, "V": 50.94, "O": 16.00, "P": 30.97,
        "S": 32.06, "F": 19.00, "Ti": 47.87, "Al": 26.98,
        "Cr": 52.00, "Zr": 91.22, "Nb": 92.91, "Li": 6.94,
        "Cu": 63.55, "Mg": 24.31,
    }

    # Use raw stoichiometric amounts to maintain mass-charge balance
    if formula:
        amounts, total = _amounts_from_formula(formula)
    else:
        # Fallback: approximate 4 atoms per formula unit
        total = 4.0
        amounts = {k: v * total for k, v in composition.items()}

    if not amounts:
        return 0.0

    # True molecular weight from raw amounts
    mw = sum(ATOMIC_MASS.get(e, 40.0) * amt for e, amt in amounts.items())
    if mw <= 0:
        return 0.0

    na_raw = amounts.get("Na", 0.0)
    if na_raw <= 0:
        return 0.0

    # Only 55% of Na is reversibly extractable in layered oxide cathodes
    # (accounts for O3→P3 phase transition structural limit)
    na_extractable = na_raw * 0.55

    # Faraday's Law: Q = (F × n_Na) / (MW × 3.6)
    capacity = (96485 * na_extractable) / (mw * 3.6)

    # Hard physical ceiling — no known SIB cathode exceeds 120 mAh/g
    return float(np.clip(capacity, 0.0, 120.0))


def _deterministic_seed(formula: str) -> float:
    h = hashlib.md5(formula.encode('utf-8')).hexdigest()
    return int(h, 16) / (16 ** 32)


def estimate_ionic_conductivity(composition: dict, formula_seed: float = 0.5) -> float:
    """Estimate ionic conductivity (S/cm)."""
    if "P" in composition and ("V" in composition or "Ti" in composition):
        return float(1e-3 + formula_seed * (1e-2 - 1e-3))
    elif "O" in composition:
        return float(1e-4 + formula_seed * (1e-3 - 1e-4))
    else:
        return float(1e-6 + formula_seed * (1e-4 - 1e-6))


def estimate_na_diffusion_barrier(composition: dict, stability: float, formula_seed: float = 0.5) -> float:
    """Estimate Na diffusion barrier (eV)."""
    base = 0.3
    if "P" in composition:
        base = 0.2
    elif "Mn" in composition:
        base = 0.35
    base += stability * 0.5
    diff = (formula_seed - 0.5) * 0.1
    return float(np.clip(base + diff, 0.1, 1.0))


def estimate_cycle_life(stability: float, formula_seed: float = 0.5, capacity_retention_target: float = 0.8) -> int:
    """Estimate cycle life (number of cycles to 80% retention)."""
    if stability < 0.02:
        return int(1500 + formula_seed * (3000 - 1500))
    elif stability < 0.05:
        return int(800 + formula_seed * (1500 - 800))
    elif stability < 0.1:
        return int(300 + formula_seed * (800 - 300))
    else:
        return int(100 + formula_seed * (300 - 100))


def compute_performance_score(
    energy_density: float,
    stability: float,
    cycle_life: int,
    ionic_conductivity: float,
) -> float:
    """Composite 0-100 performance score."""
    e_score = min(energy_density / 1000.0, 1.0) * 40
    s_score = max(0, (0.1 - stability) / 0.1) * 25
    c_score = min(cycle_life / 2000.0, 1.0) * 20
    i_score = float(np.clip((np.log10(max(ionic_conductivity, 1e-8)) + 8) / 6.0, 0.0, 1.0) * 15)
    return float(np.clip(e_score + s_score + c_score + i_score, 0, 100))


# ----------------------------------------------------------------
# FIX 1 & 7: IMPROVED CHARGE BALANCE CHECK
# Iterative (no recursion risk), tight threshold 0.05
# Uses raw stoichiometric amounts for accurate physics
# ----------------------------------------------------------------
def check_charge_balance(amounts: dict) -> tuple[float, bool]:
    """
    Check charge balance using raw stoichiometric amounts.
    Tries all physically realistic oxidation state combinations
    using itertools.product — no recursion risk.

    Args:
        amounts: dict of {element: raw_stoichiometric_amount}

    Returns:
        (best_imbalance, is_balanced)
        best_imbalance: minimum |charge| found across all oxidation state combos
        is_balanced:    True if best_imbalance <= 0.05
    """
    # All physically known oxidation states per element relevant to SIB cathodes
    STATES = {
        "Na": [1],        "Li": [1],   "K":  [1],
        "Mg": [2],        "Ca": [2],   "Al": [3],
        "Zn": [2],
        "O":  [-2],       "F":  [-1],  "S":  [-2],
        "P":  [5],        "Si": [4],
        "Fe": [2, 3],
        "Mn": [2, 3, 4],
        "Co": [2, 3],
        "Ni": [2, 3],
        "V":  [3, 4, 5],
        "Ti": [3, 4],
        "Cr": [3, 4, 6],
        "Cu": [1, 2],
        "Nb": [5],
        "Zr": [4],
        "Mo": [4, 6],
        "W":  [4, 6],
    }

    elems = list(amounts.keys())
    state_options = [STATES.get(e, [0]) for e in elems]  # Unknown → charge 0
    best = 999.0

    for combo in itertools_product(*state_options):
        charge = sum(combo[i] * amounts[elems[i]] for i in range(len(elems)))
        imbalance = abs(charge)
        if imbalance < best:
            best = imbalance
        if best <= 0.05:
            break  # Perfect balance found early

    # FIX 1: Threshold tightened 0.5 → 0.05
    return best, (best <= 0.05)


# ----------------------------------------------------------------
# FIX 3: NEW — Element consistency validator
# Detects phantom elements (in list but not formula) and
# missing elements (in formula but not in list)
# ----------------------------------------------------------------
def validate_element_consistency(formula: str, declared_elements: list) -> dict:
    """
    Cross-check formula elements against a declared elements list.

    Args:
        formula:            Chemical formula string e.g. "NaFe0.5O1.5"
        declared_elements:  List of element symbols e.g. ["Na", "Fe", "O"]

    Returns:
        dict with:
            formula_elements:  elements actually in formula
            declared_elements: elements declared in metadata
            phantom_elements:  declared but absent from formula (GAN hallucination)
            missing_elements:  in formula but not declared (GAN omission)
            is_consistent:     True only when phantom and missing are both empty
    """
    amounts, _ = _amounts_from_formula(formula)
    formula_elements = set(amounts.keys())
    declared_set = set(declared_elements)

    phantom = declared_set - formula_elements
    missing = formula_elements - declared_set

    return {
        "formula_elements": sorted(formula_elements),
        "declared_elements": sorted(declared_set),
        "phantom_elements": sorted(phantom),
        "missing_elements": sorted(missing),
        "is_consistent": len(phantom) == 0 and len(missing) == 0,
    }


# ----------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------
def validate_material(
    composition: dict, 
    formation_energy: float, 
    formula: str = "",
    declared_elements: list = None
) -> dict:
    """
    Run scientific validity checks.
    FIX 3: Now includes phantom element detection via declared_elements.
    """
    results = {}

    # 1. Formation energy < 0 for thermodynamic stability
    results["formation_energy_ok"] = formation_energy < 0

    # 2. FIX 1+3 — Charge Neutrality with raw amounts and tight threshold
    if formula:
        amounts, _ = _amounts_from_formula(formula)
    else:
        amounts = {k: v * 10 for k, v in composition.items()}

    best_imbalance, is_balanced = check_charge_balance(amounts)
    results["charge_neutral"] = is_balanced
    results["charge_imbalance"] = round(best_imbalance, 4)

    # 3. Chemical validity: must contain Na above minimum threshold
    results["chemical_validity"] = "Na" in composition and composition.get("Na", 0) > 0.05

    # 4. Redox center: must contain at least one transition metal
    tm_list = {"Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr", "Cu", "Zn", "Nb", "Zr", "Mo", "W", "Ru"}
    results["redox_active"] = any(elem in tm_list for elem in composition)

    # 5. Structural feasibility: not too many elements
    results["structural_feasibility"] = len(composition) <= 6

    # 6. Element consistency check
    if declared_elements:
        consistency = validate_element_consistency(formula, declared_elements)
        results["element_consistent"] = consistency["is_consistent"]
        results["phantom_elements"] = consistency["phantom_elements"]
        results["missing_elements"] = consistency["missing_elements"]
    else:
        results["element_consistent"] = True
        results["phantom_elements"] = []
        results["missing_elements"] = []

    # 7. Stoichiometric plausibility
    # Ensure minimum anion content (prevents O1.2 style hallucinations)
    anions = amounts.get("O", 0.0) + amounts.get("F", 0.0) + amounts.get("S", 0.0) + amounts.get("P", 0.0)
    cations = sum(amounts.get(e, 0.0) for e in ["Na", "Li", "Fe", "Mn", "Co", "Ni", "V", "Ti", "Cr", "Cu", "Zn", "Nb", "Zr"])
    
    anion_ok = anions >= 1.5
    ratio_ok = (cations / max(anions, 0.1)) <= 1.25
    results["structural_plausibility"] = anion_ok and ratio_ok

    results["passed"] = all(
        results[k] for k in [
            "formation_energy_ok", "charge_neutral",
            "chemical_validity", "redox_active", 
            "structural_feasibility", "element_consistent",
            "structural_plausibility"
        ]
    )
    return results


# ----------------------------------------------------------------
# SHAP EXPLANATION
# ----------------------------------------------------------------
def get_shap_explanation(features: np.ndarray, composition: dict) -> list:
    """Return top-k SHAP values for the band_gap model."""
    explainer = get_shap_explainer()
    shap_vals = explainer.shap_values(features.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = shap_vals.flatten()

    _, _, _, mappings = get_models()
    idx_to_elem = {v: k for k, v in mappings.get("element_to_idx", {}).items()}

    results = []
    for i, sv in enumerate(shap_vals[:62]):
        elem = idx_to_elem.get(i, f"feat_{i}")
        results.append({
            "element": elem,
            "shap_value": float(sv),
            "contribution": "positive" if sv > 0 else "negative"
        })

    present = [r for r in results if r["element"] in composition]
    absent = sorted(
        [r for r in results if r["element"] not in composition],
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )[:5]
    combined = present + absent
    combined.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return combined[:10]


def build_explanation_text(
    formula: str,
    energy_density: float,
    stability: float,
    band_gap: float,
    cycle_life: int,
    validation: dict,
) -> str:
    """Generate human-readable explanation."""
    lines = [f"Material Analysis: {formula}"]

    if energy_density > 500:
        lines.append(f" Excellent energy density ({energy_density:.0f} Wh/kg) — {energy_density/175:.1f}× commercial SIB.")
    elif energy_density > 250:
        lines.append(f" Moderate energy density ({energy_density:.0f} Wh/kg) — competitive with current SIBs.")
    else:
        lines.append(f" Low energy density ({energy_density:.0f} Wh/kg) — below commercial threshold.")

    if stability < 0.02:
        lines.append(f" Very high structural stability (Δ_hull = {stability:.4f} eV/atom).")
    elif stability < 0.1:
        lines.append(f" Acceptable stability (Δ_hull = {stability:.4f} eV/atom) — may require doping.")
    else:
        lines.append(f" Poor stability (Δ_hull = {stability:.4f} eV/atom) — not thermodynamically stable.")

    if band_gap < 1.0:
        lines.append(f" Low band gap ({band_gap:.2f} eV) — good electronic conductivity.")
    elif band_gap < 3.0:
        lines.append(f" Moderate band gap ({band_gap:.2f} eV) — may need conductive coating.")
    else:
        lines.append(f" Wide band gap ({band_gap:.2f} eV) — poor intrinsic conductivity.")

    lines.append(f"🔋 Predicted cycle life: {cycle_life} cycles (to 80% retention).")

    phantom = validation.get("phantom_elements", [])
    if phantom:
        lines.append(f"👻 **Phantom Element Alert**: Material claims to contain {', '.join(phantom)} but they are missing from the formula.")

    imbalance = validation.get("charge_imbalance", None)
    if imbalance is not None and not validation.get("charge_neutral"):
        lines.append(f"⚛️ Charge imbalance: {imbalance:.4f} (best oxidation state combo) — crystal lattice unstable.")

    if validation.get("passed"):
        lines.append("✅ **Scientifically Valid**: This composition satisfies all thermodynamic and electrochemical criteria for a sodium-ion cathode.")
    else:
        reasons = []
        if not validation.get("formation_energy_ok"):
            reasons.append("Unstable formation energy (likely to decompose)")
        if not validation.get("charge_neutral"):
            reasons.append(f"Imbalanced ionic charge (Δ={imbalance:.3f}) — crystal lattice instability")
        if not validation.get("chemical_validity"):
            reasons.append("Missing Sodium (Na) ions required for intercalation")
        if not validation.get("redox_active"):
            reasons.append("Missing Redox Center (No Transition Metal like Fe, Mn, or Ni found)")
        if not validation.get("structural_feasibility"):
            reasons.append("Composition complexity (too many elements for stable synthesis)")
        if not validation.get("element_consistent"):
            reasons.append(f"Inconsistent elements (Phantoms: {', '.join(validation.get('phantom_elements', []))})")
        if not validation.get("structural_plausibility"):
            reasons.append("Unrealistic stoichiometry (anion-poor or cation-rich structure)")
        lines.append(f"⚠️ **Validity Alert**: {'. '.join(reasons)}.")

    return " | ".join(lines)


# ----------------------------------------------------------------
# MAIN PREDICT FUNCTION
# ----------------------------------------------------------------
def predict_properties(formula: str, composition: dict = None, declared_elements: list = None) -> dict:
    """
    Full property prediction pipeline.
    Returns a dict matching PredictionResponse schema.
    """
    model_bg, model_dens, model_stab, mappings = get_models()

    if composition is None:
        composition = parse_formula(formula)

    if not composition:
        raise ValueError(f"Could not parse formula: {formula}")

    # Build feature vectors
    shared_features = build_runtime_feature_vector(formula, composition, mappings)
    descriptor_values, _ = get_descriptor_values(formula, composition)
    stability_features = build_stability_feature_vector_v2(formula, descriptor_values)

    # Core ML predictions
    pred_bg = float(model_bg.predict(shared_features.reshape(1, -1))[0])
    pred_dens = float(model_dens.predict(shared_features.reshape(1, -1))[0])

    stability_input = stability_features.reshape(1, -1)
    if hasattr(model_stab, "feature_names_in_"):
        import pandas as pd
        stability_input = pd.DataFrame([stability_features], columns=list(model_stab.feature_names_in_))
    pred_stab = float(model_stab.predict(stability_input)[0])

    formula_seed = _deterministic_seed(formula)

    # Stability model predicts formation energy directly
    formation_energy = pred_stab

    _EF_STABLE_REF = -1.5
    if formation_energy < _EF_STABLE_REF:
        structural_stability = float(np.clip(
            abs(formation_energy - _EF_STABLE_REF) * 0.01, 0.0, 0.05
        ))
    else:
        structural_stability = float(np.clip(
            (formation_energy - _EF_STABLE_REF) * 0.05 + 0.05, 0.0, 0.50
        ))

    # FIX 4: Pass formula so voltage uses raw stoichiometric amounts
    voltage = estimate_voltage(composition, pred_bg, formula=formula)
    capacity = estimate_specific_capacity(composition, pred_dens, formula=formula)

    # FIX 5: No artificial floor yet; applied conditionally below
    energy_density = voltage * capacity
    energy_density = float(np.clip(energy_density, 0, 1200))

    ionic_cond = estimate_ionic_conductivity(composition, formula_seed)
    na_barrier = estimate_na_diffusion_barrier(composition, structural_stability, formula_seed)
    cycle_life = estimate_cycle_life(structural_stability, formula_seed)
    capacity_retention = float(np.clip(0.95 - structural_stability * 2, 0.5, 0.99))

    # FIX 1 + FIX 3: Validation with tight charge balance threshold and raw amounts
    validation = validate_material(
        composition, 
        formation_energy, 
        formula=formula,
        declared_elements=declared_elements
    )

    try:
        shap_values = get_shap_explanation(shared_features, composition)
    except Exception:
        shap_values = []

    explanation = build_explanation_text(
        formula, energy_density, structural_stability, pred_bg, cycle_life, validation
    )

    score = compute_performance_score(energy_density, structural_stability, cycle_life, ionic_cond)

    if not validation.get("passed"):
        # FIX 2: Hard cap at 20 for invalid materials (was 45)
        score = float(np.clip(score * 0.10, 0, 20.0))
        # FIX 5: Cap energy density display for invalid materials — no misleading 1100 Wh/kg
        energy_density = float(np.clip(energy_density, 0, 600))
    else:
        # Valid materials: apply honest floor of 50 Wh/kg
        energy_density = float(np.clip(energy_density, 50, 1200))

    return {
        "formula": formula,
        "specific_capacity": round(capacity, 2),
        "voltage": round(voltage, 3),
        "energy_density": round(energy_density, 2),
        "formation_energy": round(formation_energy, 4),
        "ionic_conductivity": round(ionic_cond, 6),
        "na_diffusion_barrier": round(na_barrier, 4),
        "structural_stability": round(structural_stability, 4),
        "band_gap": round(pred_bg, 4),
        "density": round(pred_dens, 3),
        "cycle_life": cycle_life,
        "capacity_retention": round(capacity_retention, 4),
        "validation": validation,
        "shap_values": shap_values,
        "explanation": explanation,
        "performance_score": round(score, 2),
    }

















