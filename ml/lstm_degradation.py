"""
LSTM-based degradation simulation for sodium-ion battery cathode materials.
Predicts capacity fade and voltage decay over charge/discharge cycles.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "Models"

LSTM_HIDDEN = 64
LSTM_LAYERS = 2
INPUT_FEATURES = 5   # [cycle_norm, temp_norm, c_rate, capacity_prev_norm, voltage_prev_norm]
OUTPUT_FEATURES = 2  # [capacity, voltage]

# ADD THIS BLOCK — material-specific physics params
MATERIAL_PARAMS = {
    "NaFeO2":        (3.30, 0.15, 0.42, 0.12),
    "NaCoO2":        (3.60, 0.10, 0.32, 0.09),
    "Na2MnO3":       (3.20, 0.18, 0.48, 0.14),
    "Na3V2(PO4)3":   (3.40, 0.05, 0.14, 0.06),
    "NaNiO2":        (3.25, 0.20, 0.52, 0.15),
}

# ----------------------------------------------------------------
# LSTM Architecture
# ----------------------------------------------------------------

class DegradationLSTM(nn.Module):
    def __init__(self, input_size=INPUT_FEATURES, hidden_size=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, output_size=OUTPUT_FEATURES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden


# ----------------------------------------------------------------
# Singleton loading
# ----------------------------------------------------------------
_lstm_model: Optional[DegradationLSTM] = None


def _get_lstm():
    global _lstm_model
    if _lstm_model is not None:
        return _lstm_model

    model = DegradationLSTM()
    ckpt_path = MODEL_DIR / "lstm_degradation.pth"

    if ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
            model.eval()
            print("[LSTM] Loaded degradation checkpoint.")
        except Exception as e:
            print(f"[LSTM] Load failed ({e}), using physics-based fallback.")
    else:
        print("[LSTM] No checkpoint — will use physics-based degradation model.")

    _lstm_model = model
    return _lstm_model


# ----------------------------------------------------------------
# Physics-based degradation model (fallback + realistic curves)
# ----------------------------------------------------------------

def _physics_degradation(
    formula: str,
    initial_capacity: float,
    initial_voltage: float,
    total_cycles: int,
    stability: float,
    temperature_c: float,
    c_rate: float,
    seed: int = 42,
) -> List[dict]:
    """
    Generate realistic degradation curve using physics-informed equations.
    Combines calendar aging + cycle aging. Calibrated for 5000+ cycles.
    """
    rng = np.random.RandomState(seed)

    def _sig(x): return 1/(1+np.exp(-np.clip(x,-50,50)))

    base_v, A_sei, A_lin, V_drop_max = MATERIAL_PARAMS.get(formula, (3.30, 0.15, 0.42, 0.12))

    # stress multipliers (Arrhenius + Butler-Volmer)
    k_temp = np.exp(6000*(1/298.15 - 1/(temperature_c+273.15)))
    k_rate = 1+0.08*(c_rate-1) if c_rate>=1 else 1-0.03*(1-c_rate)
    stress = k_temp * k_rate

    A_sei_eff  = A_sei * 0.35 * stress
    A_bulk_eff = A_lin * 0.35 * stress
    A_knee_eff = A_lin * 0.30 * stress
    n_knee     = 0.40 + 0.08*(1-min(stress,2)/2)  # ~2000 cycles for N=5000
    k_knee     = 14.0
    knee_off   = _sig(k_knee*(0 - n_knee))
    V_drop_eff = V_drop_max * stress

    curve = []
    sample_points = min(total_cycles, 200)
    cycle_indices = np.linspace(0, total_cycles, sample_points, dtype=int)

    for cyc in cycle_indices:
        n = cyc / 5000.0   # ALWAYS normalize to 5000 cycles for physical consistency
        capacity_norm = float(np.clip(1 - A_sei_eff*np.sqrt(n)
                                        - A_bulk_eff*n
                                        - A_knee_eff*(_sig(k_knee*(n-n_knee)) - knee_off), 0, 1))

        capacity = initial_capacity * capacity_norm

        voltage = float(np.clip(initial_voltage - V_drop_eff * (1 - capacity_norm), 0, initial_voltage))

        deg_idx = float(np.clip(1 - capacity_norm, 0, 1))

        curve.append({
            "cycle": int(cyc),
            "capacity": round(capacity, 3),
            "voltage": round(voltage, 4),
            "degradation_index": round(deg_idx, 4),
        })

    return curve


def _find_cycle_life(curve: List[dict], initial_capacity: float, threshold=0.8) -> int:
    """Find the cycle number where capacity drops below threshold × initial."""
    target = initial_capacity * threshold
    for pt in curve:
        if pt["capacity"] <= target:
            return pt["cycle"]
    return curve[-1]["cycle"] if curve else 0


# ----------------------------------------------------------------
# LSTM inference path
# ----------------------------------------------------------------

def _lstm_degradation(
    formula: str,
    initial_capacity: float,
    initial_voltage: float,
    total_cycles: int,
    stability: float,
    temperature_c: float,
    c_rate: float,
) -> List[dict]:
    """
    Use LSTM model if checkpoint exists, else fall back to physics.
    Uses the LSTM for short-range predictions but wraps in physics envelope.
    """
    model = _get_lstm()

    sample_points = min(total_cycles, 200)
    cycle_indices = np.linspace(0, total_cycles, sample_points, dtype=int)
    curve = []

    # Normalize inputs
    temp_norm = (temperature_c - 25) / 25
    c_norm = (c_rate - 1) / 4

    capacity_prev = initial_capacity
    voltage_prev = initial_voltage
    hidden = None

    rng = np.random.RandomState(42)

    # Physics fallback blend factor (LSTM contribution fades without training)
    ckpt_exists = (MODEL_DIR / "lstm_degradation.pth").exists()
    lstm_weight = 0.1 if ckpt_exists else 0.0

    # Get physics curve for blending
    phys_curve = _physics_degradation(
        formula, initial_capacity, initial_voltage, total_cycles,
        stability, temperature_c, c_rate
    )

    if lstm_weight > 0:
        model.eval()
        with torch.no_grad():
            for i, cyc in enumerate(cycle_indices):
                cycle_norm = cyc / total_cycles
                cap_norm = capacity_prev / initial_capacity
                volt_norm = voltage_prev / initial_voltage

                inp = torch.FloatTensor([[
                    cycle_norm, temp_norm, c_norm, cap_norm, volt_norm
                ]]).unsqueeze(0)  # [1, 1, 5]

                out, hidden = model(inp, hidden)
                delta = out.squeeze().numpy()

                # delta[0] = predicted capacity, delta[1] = predicted voltage
                cap_pred = float(np.clip(delta[0] * initial_capacity, 0, initial_capacity))
                volt_pred = float(np.clip(delta[1] * initial_voltage, 0, initial_voltage))

                # Blend with physics
                phys_pt = phys_curve[i] if i < len(phys_curve) else phys_curve[-1]
                capacity = lstm_weight * cap_pred + (1 - lstm_weight) * phys_pt["capacity"]
                voltage = lstm_weight * volt_pred + (1 - lstm_weight) * phys_pt["voltage"]

                capacity_prev = capacity
                voltage_prev = voltage
                deg_idx = float(np.clip(1 - capacity / initial_capacity, 0, 1))

                curve.append({
                    "cycle": int(cyc),
                    "capacity": round(float(capacity), 3),
                    "voltage": round(float(voltage), 4),
                    "degradation_index": round(deg_idx, 4),
                })
    else:
        curve = phys_curve

    return curve


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def simulate_degradation(
    formula: str,
    initial_capacity: float = 150.0,
    total_cycles: int = 5000,
    temperature_c: float = 25.0,
    c_rate: float = 1.0,
    stability: float = 0.05,
) -> dict:
    """
    Full degradation simulation.
    Returns dict matching DegradationResponse schema.
    """
    # Generate deterministic seed for this specific formula based on physics
    import hashlib
    formula_seed = int(hashlib.md5(formula.encode('utf-8')).hexdigest(), 16) % 1000
    
    base_v, _, _, _ = MATERIAL_PARAMS.get(formula, (3.30, 0.15, 0.42, 0.12))
    initial_voltage = base_v

    curve_data = _lstm_degradation(
        formula, initial_capacity, initial_voltage, total_cycles,
        stability, temperature_c, c_rate
    )

    if not curve_data:
        raise RuntimeError("Degradation simulation produced empty curve")

    final_capacity = curve_data[-1]["capacity"]
    final_voltage = curve_data[-1]["voltage"]
    capacity_retention = float(np.clip(final_capacity / initial_capacity, 0, 1))
    voltage_decay = round(initial_voltage - final_voltage, 4)
    degradation_index = curve_data[-1]["degradation_index"]
    cycle_life = _find_cycle_life(curve_data, initial_capacity, threshold=0.8)

    # Generate summary
    if capacity_retention >= 0.8:
        health = "Excellent"
        emoji = "✅"
    elif capacity_retention >= 0.6:
        health = "Good"
        emoji = "🟡"
    else:
        health = "Poor"
        emoji = "❌"

    summary = (
        f"{emoji} {formula}: After {total_cycles} cycles, capacity retention = "
        f"{capacity_retention:.1%} ({health}). "
        f"Predicted cycle life to 80% retention: {cycle_life} cycles. "
        f"Voltage decay: {voltage_decay:.3f} V. "
        f"Degradation index: {degradation_index:.3f}."
    )

    return {
        "formula": formula,
        "total_cycles": total_cycles,
        "initial_capacity": initial_capacity,
        "final_capacity": round(final_capacity, 3),
        "capacity_retention": round(capacity_retention, 4),
        "voltage_decay": voltage_decay,
        "cycle_life_predicted": cycle_life,
        "degradation_index": round(degradation_index, 4),
        "curve_data": curve_data,
        "model_type": "LSTM+Physics" if (MODEL_DIR / "lstm_degradation.pth").exists() else "Physics-Based",
        "summary": summary,
    }
