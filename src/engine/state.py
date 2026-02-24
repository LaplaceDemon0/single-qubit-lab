from __future__ import annotations
import numpy as np

def normalize(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128).reshape(2,)
    n = np.sqrt(np.vdot(state, state).real)
    if n == 0 or not np.isfinite(n):
        raise ValueError("Cannot normalize zero/invalid state.")
    return state / n

def probs_z(state: np.ndarray) -> tuple[float, float]:
    state = np.asarray(state, dtype=np.complex128).reshape(2,)
    a, b = state[0], state[1]
    return float(abs(a) ** 2), float(abs(b) ** 2)

def bloch(state: np.ndarray) -> tuple[float, float, float]:
    state = np.asarray(state, dtype=np.complex128).reshape(2,)
    a, b = state[0], state[1]
    x = 2.0 * np.real(np.conj(a) * b)
    y = 2.0 * np.imag(np.conj(a) * b)
    z = (abs(a) ** 2) - (abs(b) ** 2)
    return float(x), float(y), float(z)