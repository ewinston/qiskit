# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum Shannon Decomposition.

Method is described in arXiv:quant-ph/0406176.
"""
import scipy
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.dagcircuit.dagnode import DAGOutNode
from qiskit.quantum_info.synthesis import two_qubit_decompose, one_qubit_decompose
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit.library.standard_gates import CZGate, CXGate
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix


class QuantumShannonDecomposer:
    """Class representation of Quantum Shannon Decomposition."""

    def __call__(self, unitary_matrix, opt_a1=True, opt_a2=False):
        return qs_decomposition(unitary_matrix, opt_a1=opt_a1, opt_a2=opt_a2)


def qs_decomposition(mat, opt_a1=True, opt_a2=False, decomposer_1q=None, decomposer_2q=None):
    """
    Decomposes unitary matrix into one and two qubit gates using Quantum Shannon Decomposition.

       ┌───┐               ┌───┐     ┌───┐     ┌───┐
      ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────
       │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐
     /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├
       └───┘          └───┘     └───┘     └───┘     └───┘

    The number of CX gates generated with the decomposition without optimizations is,

    .. math::

        \frac{9}{16} 4^n - frac{3}{2} 2^n

    If opt_a1=True, the CX count is further reduced by,

    .. math::

        \frac{1}{3} 4^{n - 2} - 1

    If opt_a2=True, the CX count is further reduced by,

    .. math::

        4^{n - 1} - 1

    Arguments:
       mat (ndarray): unitary matrix to decompose
       opt_a1 (bool): whether to try optimization A.1 from Shende. This should elliminate 1 cnot
          per call. If True CZ gates are left in the output. If desired these can be further decomposed
          to CX.
       opt_a2 (bool): whether to try optimization A.2 from Shende. Not Implemented
       decomposer_1q (None or Object): optional 1Q decomposer.
       decomposer_2q (None or Object): optional 2Q decomposer.

    Raises:
       NotImplementedError: Occurs if opt_a2=True.

    Return:
       QuantumCircuit: Decomposed quantum circuit.
    """
    dim = mat.shape[0]
    nqubits = int(np.log2(dim))
    if opt_a2:
        raise NotImplementedError("Optimization A.2 is not currently implemented.")
    if np.allclose(np.identity(dim), mat):
        return QuantumCircuit(nqubits)
    if dim == 2:
        if decomposer_1q is None:
            decomposer_1q = one_qubit_decompose.OneQubitEulerDecomposer()
        circ = decomposer_1q(mat)
    elif dim == 4:
        if decomposer_2q is None:
            decomposer_2q = two_qubit_decompose.TwoQubitBasisDecomposer(CXGate())
        circ = decomposer_2q(mat)
    else:
        qr = QuantumRegister(nqubits)
        circ = QuantumCircuit(qr)
        dim_o2 = dim // 2
        # perform cosine-sine decomposition
        (u1, u2), vtheta, (v1h, v2h) = scipy.linalg.cossin(mat, separate=True, p=dim_o2, q=dim_o2)
        # left circ
        left_circ = demultiplex(v1h, v2h, opt_a1=opt_a1, opt_a2=opt_a2)
        circ.append(left_circ.to_instruction(), qr)
        # middle circ
        if opt_a1:
            nangles = len(vtheta)
            half_size = nangles // 2
            # get UCG in terms of CZ
            circ_cz = _get_ucry_cz(nqubits, (2 * vtheta).tolist())
            circ.append(circ_cz.to_instruction(), range(nqubits))
            # merge final cz with right-side generic multiplexer
            u2[:, half_size:] = np.negative(u2[:, half_size:])
        else:
            circ.ucry((2 * vtheta).tolist(), qr[:-1], qr[-1])
        # right circ
        right_circ = demultiplex(u1, u2, opt_a1=opt_a1, opt_a2=opt_a2)
        circ.append(right_circ.to_instruction(), qr)

    return circ


def demultiplex(um0, um1, opt_a1=False, opt_a2=False):
    """decomposes a generic multiplexer.

          ────□────
           ┌──┴──┐
         /─┤     ├─
           └─────┘

    represented by the block diagonal matrix

            ┏         ┓
            ┃ um0     ┃
            ┃     um1 ┃
            ┗         ┛

    to
               ┌───┐
        ───────┤ Rz├──────
          ┌───┐└─┬─┘┌───┐
        /─┤ w ├──□──┤ v ├─
          └───┘     └───┘

    where v and w are general unitaries determined from decomposition.

    Args:
       um0 (ndarray): applied if MSB is 0
       um1 (ndarray): applied if MSB is 1
       opt_a1 (bool): whether to try optimization A.1 from Shende. This should elliminate 1 cnot
          per call. If True CZ gates are left in the output. If desired these can be further decomposed
       opt_a2 (bool): whether to try optimization A.2 from Shende. Not Implemented

          to CX.

    Returns:
        QuantumCircuit: decomposed circuit
    """
    dim = um0.shape[0] + um1.shape[0]  # these should be same dimension
    nqubits = int(np.log2(dim))
    um0um1 = um0 @ um1.T.conjugate()
    if is_hermitian_matrix(um0um1):
        eigvals, vmat = scipy.linalg.eigh(um0um1)
    else:
        evals, vmat = scipy.linalg.schur(um0um1, output="complex")
        eigvals = evals.diagonal()
    dvals = np.lib.scimath.sqrt(eigvals)
    dmat = np.diag(dvals)
    wmat = dmat @ vmat.T.conjugate() @ um1

    circ = QuantumCircuit(nqubits)

    # left gate
    left_gate = qs_decomposition(wmat, opt_a1=opt_a1, opt_a2=opt_a2).to_instruction()
    circ.append(left_gate, range(nqubits - 1))

    # multiplexed Rz
    angles = 2 * np.angle(np.conj(dvals))
    circ.ucrz(angles.tolist(), list(range(nqubits - 1)), [nqubits - 1])

    # right gate
    right_gate = qs_decomposition(vmat, opt_a1=opt_a1, opt_a2=opt_a2).to_instruction()
    circ.append(right_gate, range(nqubits - 1))

    return circ


def _get_ucry_cz(nqubits, angles):
    """
    Get uniformally controlled Ry gate in in CZ-Ry.
    """
    # avoids circular import
    from qiskit.transpiler.passes.basis.unroller import Unroller

    qc = QuantumCircuit(nqubits)
    qc.ucry(angles, list(range(nqubits - 1)), [nqubits - 1])
    dag = circuit_to_dag(qc)
    unroll = Unroller(["ry", "cx"])
    dag2 = unroll.run(dag)
    cz = CZGate()
    cxtype = type(CXGate())
    node = None
    for node in dag2.op_nodes(op=cxtype):
        dag2.substitute_node(node, cz, inplace=True)
    last_node = _get_last_op_node(dag2)
    if node.name != "cz":
        raise ValueError("last node is not cz as expected")
    dag2.remove_op_node(last_node)
    qc2 = dag_to_circuit(dag2)
    return qc2


def _get_last_op_node(dag):
    curr_node = None
    for node in dag.topological_nodes():
        if isinstance(node, DAGOutNode):
            break
        curr_node = node
    return curr_node
