from __future__ import annotations
import numpy as np

def gate_I() -> np.ndarray:
    return np.array([[1, 0], [0, 1]], dtype=np.complex128)

def gate_X() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)

def gate_Z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)

def gate_H() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

# Add more later: Y, S, T, Rx, Ry, Rz...
GATE_BUILDERS = {
    "I": gate_I,
    "X": gate_X,
    "Z": gate_Z,
    "H": gate_H,
}

def get_gate(name: str) -> np.ndarray:
    if name not in GATE_BUILDERS:
        raise KeyError(f"Unknown gate: {name}")
    return GATE_BUILDERS[name]()