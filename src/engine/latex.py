from __future__ import annotations
import numpy as np

def _fmt_c(z: complex, nd: int = 4) -> str:
    """Format complex as LaTeX-friendly a+bi."""
    re = float(np.round(z.real, nd))
    im = float(np.round(z.imag, nd))

    if abs(im) < 1e-12:
        return f"{re:g}"
    if abs(re) < 1e-12:
        return f"{im:g}i"
    sign = "+" if im >= 0 else "-"
    return f"{re:g} {sign} {abs(im):g}i"

def vec_latex(state: np.ndarray, nd: int = 4) -> str:
    state = np.asarray(state, dtype=np.complex128).reshape(2,)
    a, b = state[0], state[1]
    return rf"\begin{{bmatrix}} {_fmt_c(a, nd)} \\ {_fmt_c(b, nd)} \end{{bmatrix}}"

def mat_latex(U: np.ndarray, nd: int = 4) -> str:
    U = np.asarray(U, dtype=np.complex128).reshape(2, 2)
    return (
        rf"\begin{{bmatrix}} "
        rf"{_fmt_c(U[0,0], nd)} & {_fmt_c(U[0,1], nd)} \\ "
        rf"{_fmt_c(U[1,0], nd)} & {_fmt_c(U[1,1], nd)} "
        rf"\end{{bmatrix}}"
    )