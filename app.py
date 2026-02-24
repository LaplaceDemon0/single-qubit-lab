import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st

from src.engine import Circuit1Q, get_gate, GATE_BUILDERS
from src.engine.state import normalize, probs_z, bloch
from src.engine.viz import draw_circuit_1q, plot_probs, plot_bloch_point
from src.engine.latex import mat_latex, vec_latex


st.set_page_config(page_title="cipherQ single qubit lab", layout="wide")
st.title("cipherQ single qubit lab")


if "gate_names" not in st.session_state:
    st.session_state.gate_names = []


topL, topR = st.columns([1, 1], gap="large")

with topL:
    st.subheader("Inputs")

    preset = st.selectbox(
        "Initial state",
        ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "Custom (α, β)"],
        index=0
    )

    if preset == "|0⟩":
        state0 = np.array([1+0j, 0+0j], dtype=np.complex128)
    elif preset == "|1⟩":
        state0 = np.array([0+0j, 1+0j], dtype=np.complex128)
    elif preset == "|+⟩":
        state0 = normalize(get_gate("H") @ np.array([1+0j, 0+0j], dtype=np.complex128))
    elif preset == "|−⟩":
        state0 = normalize(get_gate("H") @ np.array([0+0j, 1+0j], dtype=np.complex128))
    else:
        c1, c2 = st.columns(2)
        with c1:
            a_re = st.number_input("α real", value=1.0)
            a_im = st.number_input("α imag", value=0.0)
        with c2:
            b_re = st.number_input("β real", value=0.0)
            b_im = st.number_input("β imag", value=0.0)

        state0 = normalize(np.array([a_re + 1j*a_im, b_re + 1j*b_im], dtype=np.complex128))

    st.markdown("### Build circuit")

    gate_to_add = st.selectbox("Add gate", list(GATE_BUILDERS.keys()), index=3)

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("Add gate"):
            st.session_state.gate_names.append(gate_to_add)
    with b2:
        if st.button("Undo") and st.session_state.gate_names:
            st.session_state.gate_names.pop()
    with b3:
        if st.button("Clear"):
            st.session_state.gate_names = []

    st.write("Sequence:", " → ".join(st.session_state.gate_names) if st.session_state.gate_names else "(empty)")



circuit = Circuit1Q("q0")
for name in st.session_state.gate_names:
    circuit.add(name, get_gate(name))

chain = circuit.run(state0)
final_state = chain[-1][1]
p0, p1 = probs_z(final_state)
bx, by, bz = bloch(final_state)


with topR:
    st.subheader("Circuit")
    st.pyplot(draw_circuit_1q(st.session_state.gate_names, label="q0"), use_container_width=True)


st.divider()



midL, midR = st.columns([1, 1], gap="large")

with midL:
    st.subheader("Measurement probabilities (Z basis)")
    st.pyplot(plot_probs(p0, p1), use_container_width=True)

with midR:
    st.subheader("Bloch sphere")
    st.pyplot(plot_bloch_point(bx, by, bz), use_container_width=True)


st.divider()



botL, botR = st.columns([1, 1], gap="large")

with botL:
    st.subheader("Math (step-by-step)")

    show_math = st.checkbox("Show matrices and steps", value=True)
    if show_math:
        
        st.latex(rf"|\psi_0\rangle = {vec_latex(chain[0][1])}")

        
        st.latex(rf"|\psi_0\rangle = {vec_latex(chain[0][1])}")

        for i in range(1, len(chain)):
            gate_name = chain[i][0]
            psi_prev = chain[i-1][1]
            psi_now = chain[i][1]
            U = get_gate(gate_name)

            st.latex(rf"U_{i} = {gate_name} = {mat_latex(U)}")
            st.latex(
                rf"|\psi_{i}\rangle = {gate_name}|\psi_{i-1}\rangle"
                rf"= {mat_latex(U)} {vec_latex(psi_prev)}"
            )
            st.latex(rf"|\psi_{i}\rangle = {vec_latex(psi_now)}")

        
        if st.session_state.gate_names:
            op_sequence = " ".join(st.session_state.gate_names)

            st.markdown("### Operator Composition")
            st.latex(rf"|\psi_{{final}}\rangle = {op_sequence}\,|\psi_0\rangle")
            st.latex(rf"|\psi_{{final}}\rangle = {vec_latex(final_state)}")
        else:
            st.info("Enable the checkbox to show the full matrix steps.")

with botR:
    st.subheader("Final results")

    st.markdown("**Final state (numeric):**")
    st.write(final_state)

    st.markdown("**Probabilities (Z basis):**")
    st.write(f"P(0) = {p0:.6f}")
    st.write(f"P(1) = {p1:.6f}")

    st.markdown("**Bloch coordinates:**")
    st.write(f"(x, y, z) = ({bx:.6f}, {by:.6f}, {bz:.6f})")

    st.markdown("**Checks:**")
    st.write("⟨ψ|ψ⟩ =", float(np.vdot(final_state, final_state).real))
    st.write("Gate count =", len(st.session_state.gate_names))