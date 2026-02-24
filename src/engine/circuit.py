from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .state import normalize

@dataclass(frozen=True)
class GateOp:
    name: str
    U: np.ndarray

class Circuit1Q:
    def __init__(self, label: str = "q0"):
        self.label = label
        self.ops: list[GateOp] = []

    def add(self, name: str, U: np.ndarray) -> "Circuit1Q":
        U = np.asarray(U, dtype=np.complex128)
        if U.shape != (2, 2):
            raise ValueError("Gate must be 2x2.")
        self.ops.append(GateOp(name=name, U=U))
        return self

    def pop(self) -> None:
        if self.ops:
            self.ops.pop()

    def clear(self) -> None:
        self.ops = []

    def run(self, state0: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """
        Returns step chain: [("Input", |psi0>), ("H", |psi1>), ...]
        """
        state = normalize(state0)
        chain: list[tuple[str, np.ndarray]] = [("Input", state.copy())]
        for op in self.ops:
            state = normalize(op.U @ state)
            chain.append((op.name, state.copy()))
        return chain