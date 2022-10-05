# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Stochastic Swap pass"""

import unittest
from ddt import ddt, data
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.utils import CheckMap


@ddt
class TestStochasticSwap(QiskitTestCase):
    """
    Tests the StochasticSwap pass.
    All of the tests use a fixed seed since the results
    may depend on it.
    """

    def test_trivial_case(self):
        """
        q0:--(+)-[H]-(+)-
              |       |
        q1:---.-------|--
                      |
        q2:-----------.--
        Coupling map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_trivial_in_same_layer(self):
        """
        q0:--(+)--
              |
        q1:---.---
        q2:--(+)--
              |
        q3:---.---
        Coupling map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_1(self):
        """
        q0:--------

        q1:---.----
              |
        q2:--(+)---
        Coupling map: [1]--[0]--[2]
        q0:--x-(+)-
             |  |
        q1:--|--.--
             |
        q2:--x-----
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 11)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_2(self):
        """
        qr0:---.---[H]--
               |
        qr1:---|--------
               |
        qr2:--(+)-------
        Coupling map: [0]--[1]--[2]
        qr0:----.---[H]-
                |
        qr1:-x-(+)------
             |
        qr2:-x----------
        """
        coupling = CouplingMap([[1, 0], [1, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 11)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[2])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])

        self.assertEqual(expected, dag_to_circuit(after))

    def test_permute_wires_3(self):
        """
        qr0:--(+)---.--
               |    |
        qr1:---|----|--
               |    |
        qr2:---|----|--
               |    |
        qr3:---.---(+)-
        Coupling map: [0]--[1]--[2]--[3]
        qr0:-x------------
             |
        qr1:-x--(+)---.---
                 |    |
        qr2:-x---.---(+)--
             |
        qr3:-x------------
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.cx(qr[2], qr[1])

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_4(self):
        """No qubit label permutation occurs if the first
        layer has only single-qubit gates. This is suboptimal
        but seems to be the current behavior.
         qr0:------(+)--
                    |
         qr1:-------|---
                    |
         qr2:-------|---
                    |
         qr3:--[H]--.---
         Coupling map: [0]--[1]--[2]--[3]
         qr0:------X---------
                   |
         qr1:------X-(+)-----
                      |
         qr2:------X--.------
                   |
         qr3:-[H]--X---------
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[2], qr[3])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_5(self):
        """This is the same case as permute_wires_4
        except the single qubit gate is after the two-qubit
        gate, so the layout is adjusted.
         qr0:--(+)------
                |
         qr1:---|-------
                |
         qr2:---|-------
                |
         qr3:---.--[H]--
         Coupling map: [0]--[1]--[2]--[3]
         qr0:-x-----------
              |
         qr1:-x--(+)------
                  |
         qr2:-x---.--[H]--
              |
         qr3:-x-----------
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[2], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])

        self.assertEqual(circuit_to_dag(expected), after)

    def test_all_single_qubit(self):
        """Test all trivial layers."""
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circ = QuantumCircuit(qr, cr)
        circ.h(qr)
        circ.z(qr)
        circ.s(qr)
        circ.t(qr)
        circ.tdg(qr)
        circ.measure(qr[0], cr[0])  # intentional duplicate
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])
        dag = circuit_to_dag(circ)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_overoptimization_case(self):
        """Check mapper overoptimization.
        The mapper should not change the semantics of the input.
        An overoptimization introduced issue #81:
        https://github.com/Qiskit/qiskit-terra/issues/81
        """
        coupling = CouplingMap([[0, 2], [1, 2], [2, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.y(qr[1])
        circuit.z(qr[2])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.s(qr[1])
        circuit.t(qr[2])
        circuit.h(qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[2], cr[2])
        circuit.measure(qr[3], cr[3])
        dag = circuit_to_dag(circuit)
        #           ┌───┐                                        ┌─┐
        # q_0: | 0 >┤ X ├────────────■───────────────────────────┤M├─────────
        #           └───┘┌───┐     ┌─┴─┐     ┌───┐               └╥┘┌─┐
        # q_1: | 0 >─────┤ Y ├─────┤ X ├─────┤ S ├────────────■───╫─┤M├──────
        #                └───┘┌───┐└───┘     └───┘┌───┐     ┌─┴─┐ ║ └╥┘┌─┐
        # q_2: | 0 >──────────┤ Z ├───────■───────┤ T ├─────┤ X ├─╫──╫─┤M├───
        #                     └───┘     ┌─┴─┐     └───┘┌───┐└───┘ ║  ║ └╥┘┌─┐
        # q_3: | 0 >────────────────────┤ X ├──────────┤ H ├──────╫──╫──╫─┤M├
        #                               └───┘          └───┘      ║  ║  ║ └╥┘
        # c_0: 0    ══════════════════════════════════════════════╩══╬══╬══╬═
        #                                                            ║  ║  ║
        #  c_1: 0   ═════════════════════════════════════════════════╩══╬══╬═
        #                                                               ║  ║
        #  c_2: 0   ════════════════════════════════════════════════════╩══╬═
        #                                                                  ║
        #  c_3: 0   ═══════════════════════════════════════════════════════╩═
        #
        expected = QuantumCircuit(qr, cr)
        expected.z(qr[2])
        expected.y(qr[1])
        expected.x(qr[0])
        expected.swap(qr[0], qr[2])
        expected.cx(qr[2], qr[1])
        expected.swap(qr[0], qr[2])
        expected.cx(qr[2], qr[3])
        expected.s(qr[1])
        expected.t(qr[2])
        expected.h(qr[3])
        expected.measure(qr[0], cr[0])
        expected.cx(qr[1], qr[2])
        expected.measure(qr[3], cr[3])
        expected.measure(qr[1], cr[1])
        expected.measure(qr[2], cr[2])
        expected_dag = circuit_to_dag(expected)
        #      ┌───┐                ┌─┐
        # q_0: ┤ X ├─X───────X──────┤M├────────────────
        #      ├───┤ │ ┌───┐ │ ┌───┐└╥┘          ┌─┐
        # q_1: ┤ Y ├─┼─┤ X ├─┼─┤ S ├─╫────────■──┤M├───
        #      ├───┤ │ └─┬─┘ │ └───┘ ║ ┌───┐┌─┴─┐└╥┘┌─┐
        # q_2: ┤ Z ├─X───■───X───■───╫─┤ T ├┤ X ├─╫─┤M├
        #      └───┘           ┌─┴─┐ ║ ├───┤└┬─┬┘ ║ └╥┘
        # q_3: ────────────────┤ X ├─╫─┤ H ├─┤M├──╫──╫─
        #                      └───┘ ║ └───┘ └╥┘  ║  ║
        # c: 4/══════════════════════╩════════╩═══╩══╩═
        #                            0        3   1  2

        #
        # Layout --
        #  {qr[0]: 0,
        #  qr[1]: 1,
        #  qr[2]: 2,
        #  qr[3]: 3}
        pass_ = StochasticSwap(coupling, 20, 19)
        after = pass_.run(dag)

        self.assertEqual(expected_dag, after)

    def test_already_mapped(self):
        """Circuit not remapped if matches topology.
        See: https://github.com/Qiskit/qiskit-terra/issues/342
        """
        coupling = CouplingMap(
            [
                [1, 0],
                [1, 2],
                [2, 3],
                [3, 4],
                [3, 14],
                [5, 4],
                [6, 5],
                [6, 7],
                [6, 11],
                [7, 10],
                [8, 7],
                [9, 8],
                [9, 10],
                [11, 10],
                [12, 5],
                [12, 11],
                [12, 13],
                [13, 4],
                [13, 14],
                [15, 0],
                [15, 0],
                [15, 2],
                [15, 14],
            ]
        )
        qr = QuantumRegister(16, "q")
        cr = ClassicalRegister(16, "c")
        circ = QuantumCircuit(qr, cr)
        circ.cx(qr[3], qr[14])
        circ.cx(qr[5], qr[4])
        circ.h(qr[9])
        circ.cx(qr[9], qr[8])
        circ.x(qr[11])
        circ.cx(qr[3], qr[4])
        circ.cx(qr[12], qr[11])
        circ.cx(qr[13], qr[4])
        for j in range(16):
            circ.measure(qr[j], cr[j])

        dag = circuit_to_dag(circ)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(circ), after)

    def test_congestion(self):
        """Test code path that falls back to serial layers."""
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circ = QuantumCircuit(qr, cr)
        circ.cx(qr[1], qr[2])
        circ.cx(qr[0], qr[3])
        circ.measure(qr[0], cr[0])
        circ.h(qr)
        circ.cx(qr[0], qr[1])
        circ.cx(qr[2], qr[3])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])
        dag = circuit_to_dag(circ)
        # Input:
        #                                             ┌─┐┌───┐        ┌─┐
        # q_0: |0>─────────────────■──────────────────┤M├┤ H ├──■─────┤M├
        #                   ┌───┐  │                  └╥┘└───┘┌─┴─┐┌─┐└╥┘
        # q_1: |0>──■───────┤ H ├──┼───────────────────╫──────┤ X ├┤M├─╫─
        #         ┌─┴─┐┌───┐└───┘  │               ┌─┐ ║      └───┘└╥┘ ║
        # q_2: |0>┤ X ├┤ H ├───────┼─────────■─────┤M├─╫────────────╫──╫─
        #         └───┘└───┘     ┌─┴─┐┌───┐┌─┴─┐┌─┐└╥┘ ║            ║  ║
        # q_3: |0>───────────────┤ X ├┤ H ├┤ X ├┤M├─╫──╫────────────╫──╫─
        #                        └───┘└───┘└───┘└╥┘ ║  ║            ║  ║
        #  c_0: 0 ═══════════════════════════════╬══╬══╩════════════╬══╩═
        #                                        ║  ║               ║
        #  c_1: 0 ═══════════════════════════════╬══╬═══════════════╩════
        #                                        ║  ║
        #  c_2: 0 ═══════════════════════════════╬══╩════════════════════
        #                                        ║
        #  c_3: 0 ═══════════════════════════════╩═══════════════════════
        #
        # Expected output (with seed 999):
        #                ┌───┐                        ┌─┐
        # q_0: ───────X──┤ H ├─────────────────X──────┤M├──────
        #             │  └───┘     ┌─┐   ┌───┐ │ ┌───┐└╥┘   ┌─┐
        # q_1: ──■────X────■───────┤M├─X─┤ X ├─X─┤ X ├─╫────┤M├
        #      ┌─┴─┐┌───┐  │       └╥┘ │ └─┬─┘┌─┐└─┬─┘ ║    └╥┘
        # q_2: ┤ X ├┤ H ├──┼────────╫──┼───■──┤M├──┼───╫─────╫─
        #      └───┘└───┘┌─┴─┐┌───┐ ║  │ ┌───┐└╥┘  │   ║ ┌─┐ ║
        # q_3: ──────────┤ X ├┤ H ├─╫──X─┤ H ├─╫───■───╫─┤M├─╫─
        #                └───┘└───┘ ║    └───┘ ║       ║ └╥┘ ║
        # c: 4/═════════════════════╩══════════╩═══════╩══╩══╩═
        #                           0          2       3  0  1
        #
        # Target coupling graph:
        #     2
        #     |
        # 0 - 1 - 3

        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[1], qr[2])
        expected.h(qr[2])
        expected.swap(qr[0], qr[1])
        expected.h(qr[0])
        expected.cx(qr[1], qr[3])
        expected.h(qr[3])
        expected.measure(qr[1], cr[0])
        expected.swap(qr[1], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[3])
        expected.swap(qr[0], qr[1])
        expected.measure(qr[2], cr[2])
        expected.cx(qr[3], qr[1])
        expected.measure(qr[0], cr[3])
        expected.measure(qr[3], cr[0])
        expected.measure(qr[1], cr[1])
        expected_dag = circuit_to_dag(expected)

        pass_ = StochasticSwap(coupling, 20, 999)
        after = pass_.run(dag)
        self.assertEqual(expected_dag, after)

    def test_only_output_cx_and_swaps_in_coupling_map(self):
        """Test that output DAG contains only 2q gates from the the coupling map."""

        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 5)
        after = pass_.run(dag)

        valid_couplings = [{qr[a], qr[b]} for (a, b) in coupling.get_edges()]

        for _2q_gate in after.two_qubit_ops():
            self.assertIn(set(_2q_gate.qargs), valid_couplings)

    def test_len_cm_vs_dag(self):
        """Test error if the coupling map is smaller than the dag."""

        coupling = CouplingMap([[0, 1], [1, 2]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling)
        with self.assertRaises(TranspilerError):
            _ = pass_.run(dag)

    def test_single_gates_omitted(self):
        """Test if single qubit gates are omitted."""

        coupling_map = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]

        # q_0: ──■──────────────────
        #        │
        # q_1: ──┼─────────■────────
        #        │       ┌─┴─┐
        # q_2: ──┼───────┤ X ├──────
        #        │  ┌────┴───┴─────┐
        # q_3: ──┼──┤ U(1,1.5,0.7) ├
        #      ┌─┴─┐└──────────────┘
        # q_4: ┤ X ├────────────────
        #      └───┘
        qr = QuantumRegister(5, "q")
        cr = ClassicalRegister(5, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[4])
        circuit.cx(qr[1], qr[2])
        circuit.u(1, 1.5, 0.7, qr[3])

        # q_0: ─────────────────X──────
        #                       │
        # q_1: ───────■─────────X───■──
        #           ┌─┴─┐           │
        # q_2: ─────┤ X ├───────────┼──
        #      ┌────┴───┴─────┐   ┌─┴─┐
        # q_3: ┤ U(1,1.5,0.7) ├─X─┤ X ├
        #      └──────────────┘ │ └───┘
        # q_4: ─────────────────X──────
        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[1], qr[2])
        expected.u(1, 1.5, 0.7, qr[3])
        expected.swap(qr[0], qr[1])
        expected.swap(qr[3], qr[4])
        expected.cx(qr[1], qr[3])

        expected_dag = circuit_to_dag(expected)

        stochastic = StochasticSwap(CouplingMap(coupling_map), seed=0)
        after = PassManager(stochastic).run(circuit)
        after = circuit_to_dag(after)
        self.assertEqual(expected_dag, after)

    def test_controlflow_pre_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=82).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(0, 1)
        expected.cx(1, 2)
        expected.measure(2, 2)
        eqreg = QuantumRegister(2, name="q")
        etrue_body = QuantumCircuit(eqreg, creg)
        etrue_body.x(0)
        efalse_body = QuantumCircuit(eqreg, creg)
        efalse_body.x(1)
        new_order = [1, 0, 2, 3, 4]
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg[[3, 4]], creg)
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_pre_if_else_route_post_x(self):
        """test swap with if else controlflow construct; pre-cx and post x"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg)
        qc.x(1)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=431).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(1, 2)
        new_order = [0, 2, 1, 3, 4]
        eqreg = QuantumRegister(2, "q")
        etrue_body = QuantumCircuit(eqreg, creg)
        etrue_body.x(0)
        efalse_body = QuantumCircuit(eqreg, creg)
        efalse_body.x(1)
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg[[3, 4]], creg)
        expected.x(2)
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])

        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_post_if_else_route(self):
        """test swap with if else controlflow construct; post cx"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.barrier(qreg)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.barrier(qreg)
        qc.cx(0, 2)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=6508).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.measure(0, 0)
        eqreg = QuantumRegister(2, "q")
        etrue_body = QuantumCircuit(eqreg, creg)
        etrue_body.x(0)
        efalse_body = QuantumCircuit(eqreg, creg)
        efalse_body.x(1)
        expected.barrier(qreg)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[3, 4]], creg)
        expected.barrier(qreg)
        expected.swap(0, 1)
        expected.cx(1, 2)
        expected.barrier(qreg)
        expected.measure(qreg, creg[[1, 0, 2, 3, 4]])
        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_pre_if_else2(self):
        """test swap with if else controlflow construct; cx in if statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(0)
        false_body = QuantumCircuit(qreg, creg)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=38).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(0, 1)
        expected.cx(1, 2)
        expected.measure(1, 0)
        eqreg = QuantumRegister(1, "q")
        etrue_body = QuantumCircuit(eqreg, creg)
        etrue_body.x(0)
        efalse_body = QuantumCircuit(eqreg, creg)
        new_order = [1, 0, 2, 3, 4]
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[1]], creg)
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])

        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_intra_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=21).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg)
        etrue_body.swap(0, 1)
        etrue_body.cx(1, 2)
        etrue_body.swap(1, 2)
        etrue_body.swap(3, 4)
        efalse_body = QuantumCircuit(qreg, creg)
        efalse_body.swap(0, 1)
        efalse_body.swap(1, 2)
        efalse_body.swap(3, 4)
        efalse_body.cx(2, 3)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg)
        new_order = [1, 2, 0, 4, 3]
        expected.measure(qreg, creg[new_order])

        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    @data(1, 2)
    def test_controlflow_pre_intra_if_else(self, seed):
        """test swap with if else controlflow construct; cx in if statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=seed, trials=20).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        if seed == 1:
            eqreg = QuantumRegister(5, "q")
            etrue_body = QuantumCircuit(eqreg, creg)
            efalse_body = QuantumCircuit(eqreg, creg)
            expected.h(0)
            expected.x(1)
            expected.swap(2, 1)
            expected.cx(0, 1)
            expected.measure(0, 0)

            etrue_body.cx(0, 2)
            etrue_body.swap(0, 2)
            etrue_body.swap(2, 1)
            etrue_body.swap(4, 3)

            efalse_body.swap(0, 2)
            efalse_body.swap(2, 1)
            efalse_body.swap(3, 4)
            efalse_body.cx(1, 3)
            expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[0, 2, 1, 3, 4]], creg)
            expected.measure(qreg, creg[[2, 1, 0, 4, 3]])
        elif seed == 2:
            eqreg = QuantumRegister(4, "q")
            etrue_body = QuantumCircuit(eqreg, creg)
            efalse_body = QuantumCircuit(eqreg, creg)
            expected.h(0)
            expected.x(1)
            expected.swap(0, 1)
            expected.cx(1, 2)
            expected.measure(1, 0)

            etrue_body.cx(0, 1)
            etrue_body.swap(2, 3)
            etrue_body.swap(0, 1)

            efalse_body.swap(0, 1)
            efalse_body.swap(2, 3)
            efalse_body.cx(1, 2)
            expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[1, 2, 3, 4]], creg)
            expected.measure(qreg, creg[[1, 2, 0, 4, 3]])
        else:
            raise ValueError(f"unsupported test seed={seed}")

        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_pre_intra_post_if_else(self):
        """test swap with if else controlflow construct; cx before, in, and after if
        statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.h(3)
        qc.cx(3, 0)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=58).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg)
        etrue_body.cx(0, 2)
        etrue_body.swap(0, 2)
        etrue_body.swap(4, 3)
        etrue_body.swap(1, 3)
        efalse_body = QuantumCircuit(qreg, creg)
        efalse_body.swap(0, 2)
        efalse_body.swap(3, 4)
        efalse_body.swap(1, 3)
        efalse_body.cx(2, 1)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[0, 2, 1, 3, 4]], creg)
        expected.swap(1, 2)
        expected.h(4)
        expected.swap(3, 4)
        expected.cx(3, 2)
        expected.barrier()
        expected.measure(qreg, creg[[2, 4, 0, 3, 1]])

        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_no_layout_change(self):
        """test controlflow with no layout change needed"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(2)
        false_body = QuantumCircuit(qreg, creg)
        false_body.x(4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=23).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        eqreg = QuantumRegister(2, "q")
        etrue_body = QuantumCircuit(eqreg, creg)
        etrue_body.x(0)
        efalse_body = QuantumCircuit(eqreg, creg)
        efalse_body.x(1)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[1, 4]], creg)
        expected.barrier(qreg)
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])

        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    @data(1, 2, 3)
    def test_controlflow_for_loop(self, nloops):
        """test stochastic swap with for_loop"""
        # if the loop has only one iteration it isn't necessary for the pass
        # to swap back to the starting layout. This test would check that
        # optimization.
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        for_body = QuantumCircuit(qreg, creg)
        for_body.cx(0, 2)
        loop_parameter = None
        qc.for_loop(range(nloops), loop_parameter, for_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=687).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        efor_body = QuantumCircuit(qreg, creg)
        efor_body.swap(0, 1)
        efor_body.cx(1, 2)
        efor_body.swap(0, 1)
        loop_parameter = None
        expected.for_loop(range(nloops), loop_parameter, efor_body, qreg, creg)
        expected.measure(qreg, creg)
        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_while_loop(self):
        """test while loop"""
        num_qubits = 4
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(len(qreg))
        coupling = CouplingMap.from_line(num_qubits)
        check_map_pass = CheckMap(coupling)
        qc = QuantumCircuit(qreg, creg)
        while_body = QuantumCircuit(qreg, creg)
        while_body.reset(qreg[2:])
        while_body.h(qreg[2:])
        while_body.cx(0, 3)
        while_body.measure(qreg[3], creg[3])
        qc.while_loop((creg, 0), while_body, qc.qubits, qc.clbits)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=58).run(dag)
        cqc = dag_to_circuit(cdag)
        expected = QuantumCircuit(qreg, creg)
        ewhile_body = QuantumCircuit(qreg, creg)
        ewhile_body.reset(qreg[2:])
        ewhile_body.h(qreg[2:])
        ewhile_body.swap(0, 1)
        ewhile_body.swap(2, 3)
        ewhile_body.cx(1, 2)
        ewhile_body.measure(qreg[2], creg[3])
        ewhile_body.swap(1, 0)
        ewhile_body.swap(3, 2)
        expected.while_loop((creg, 0), ewhile_body, expected.qubits, expected.clbits)
        expected.barrier()
        expected.measure(qreg, creg)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertEqual(cqc, expected)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_nested_inner_cnot(self):
        """test swap in nested if else controlflow construct; swap in inner"""
        seed = 392
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        check_map_pass = CheckMap(coupling)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.x(0)

        for_body = QuantumCircuit(qreg, creg)
        for_body.delay(10, 0)
        for_body.barrier(qreg)
        for_body.cx(0, 2)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, creg)

        false_body = QuantumCircuit(qreg, creg)
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=seed).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg)
        etrue_body.x(0)

        efor_body = QuantumCircuit(qreg, creg)
        efor_body.delay(10, 0)
        efor_body.barrier(qreg)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        etrue_body.for_loop(range(3), loop_parameter, efor_body, qreg, creg)

        efalse_body = QuantumCircuit(qreg, creg)
        efalse_body.y(0)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg)
        expected.measure(qreg, creg)

        self.assertEqual(cqc, expected)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_controlflow_nested_outer_cnot(self):
        """test swap with nested if else controlflow construct; swap in outer"""
        seed = 200
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        check_map_pass = CheckMap(coupling)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg)
        true_body.cx(0, 2)
        true_body.x(0)

        for_body = QuantumCircuit(qreg, creg)
        for_body.delay(10, 0)
        for_body.barrier(qreg)
        for_body.cx(1, 3)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, creg)

        false_body = QuantumCircuit(qreg, creg)
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = StochasticSwap(coupling, seed=seed).run(dag)
        cqc = dag_to_circuit(cdag)

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg)
        etrue_body.swap(1, 2)
        etrue_body.cx(0, 1)
        etrue_body.x(0)

        efor_body = QuantumCircuit(qreg, creg)
        efor_body.delay(10, 0)
        efor_body.barrier(qreg)
        efor_body.cx(1, 3)
        etrue_body.for_loop(range(3), loop_parameter, efor_body, qreg[[0, 2, 1, 3, 4]], creg)

        efalse_body = QuantumCircuit(qreg, creg)
        efalse_body.y(0)
        efalse_body.swap(1, 2)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg)
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])

        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])
        self.assertEqual(cqc, expected)

    def test_controlflow_disjoint_looping(self):
        """Test looping controlflow on different qubit register"""
        num_qubits = 4
        cm = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)
        loop_body = QuantumCircuit(2)
        loop_body.cx(0, 1)
        qc.for_loop((0,), None, loop_body, [0, 2], [])
        cqc = StochasticSwap(cm, seed=653)(qc)

        expected = QuantumCircuit(qr)
        efor_body = QuantumCircuit(3)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        expected.for_loop((0,), None, efor_body, [0, 1, 2], [])
        self.assertEqual(cqc, expected)

    def test_controlflow_disjoint_multiblock(self):
        """Test looping controlflow on different qubit register"""
        num_qubits = 4
        cm = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(3, 1)
        true_body.cx(0, 1)
        false_body = QuantumCircuit(3, 1)
        false_body.cx(0, 2)
        qc.if_else((cr[0], 1), true_body, false_body, [0, 1, 2], [])
        cqc = StochasticSwap(cm, seed=353)(qc)

        expected = QuantumCircuit(qr, cr)
        etrue_body = QuantumCircuit(true_body.qregs[0], cr)
        etrue_body.cx(0, 1)
        etrue_body.swap(0, 1)
        efalse_body = QuantumCircuit(false_body.qregs[0], cr)
        efalse_body.swap(0, 1)
        efalse_body.cx(1, 2)
        expected.if_else((cr[0], 1), etrue_body, efalse_body, [0, 1, 2], [])
        self.assertEqual(cqc, expected)

    def test_controlflow_multiple_ops_per_layer(self):
        """Test circuits with multiple operations per layer"""
        num_qubits = 5
        coupling = CouplingMap.from_line(num_qubits)
        check_map_pass = CheckMap(coupling)
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 2)
        with qc.for_loop((0,)):
            qc.cx(2, 4)
        cqc = StochasticSwap(coupling, seed=68745)(qc)

        expected = QuantumCircuit(qr)
        expected.swap(0, 1)
        expected.cx(1, 2)
        efor_body = QuantumCircuit(3)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(2, 1)
        expected.for_loop((0,), None, efor_body, [2, 3, 4], [])

        check_map_pass.run(circuit_to_dag(expected))
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])
        self.assertEqual(cqc, expected)


if __name__ == "__main__":
    unittest.main()
