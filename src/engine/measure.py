from __future__ import annotations
import numpy as np

def measure_z(state: np.ndarray, rng: np.random.Generator | None = None) -> tuple[int, np.ndarray]:
    """
    Measure in Z basis, return (outcome, collapsed_state).
    """
    rng = rng or np.random.default_rng()
    state = np.asarray(state, dtype=np.complex128).reshape(2,)
    a, b = state[0], state[1]
    p0 = float(abs(a) ** 2)
    r = float(rng.random())
    if r < p0:
        return 0, np.array([1+0j, 0+0j], dtype=np.complex128)
    return 1, np.array([0+0j, 1+0j], dtype=np.complex128)