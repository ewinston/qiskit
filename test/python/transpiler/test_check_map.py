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

"""Test the Check Map pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestCheckMapCX(QiskitTestCase):
    """Tests the CheckMap pass with CX gates"""

    def test_trivial_nop_map(self):
        """Trivial map in a circuit without entanglement
        qr0:---[H]---

        qr1:---[H]---

        qr2:---[H]---

        CouplingMap map: None
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_true(self):
        """Mapped is easy to check
        qr0:--(+)-[H]-(+)-
               |       |
        qr1:---.-------|--
                       |
        qr2:-----------.--

        CouplingMap map: [1]--[0]--[2]
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_false(self):
        """Needs [0]-[1] in a [0]--[2]--[1]
        qr0:--(+)--
               |
        qr1:---.---

        CouplingMap map: [0]--[2]--[1]
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 2], [2, 1]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_true(self):
        """Check control flow blocks are mapped."""
        num_qubits = 3
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr)
        true_body.swap(0, 1)
        true_body.cx(2, 1)
        circuit.if_else((cr[0], 0), true_body, None, qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_false(self):
        """Check control flow blocks are not mapped."""
        num_qubits = 3
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr)
        true_body.cx(0, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_layout_change_false(self):
        """Check control flow blocks with layout change are not mapped."""
        num_qubits = 3
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr)
        true_body.cx(1, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr[[1, 0, 2]], cr)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        print(circuit)
        print(circuit.data[0].operation.params[0])
        print(pass_.property_set["is_swap_mapped"])
        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_layout_change_true(self):
        """Check control flow blocks with layout change are mapped."""
        num_qubits = 3
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr)
        true_body.cx(0, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr[[1, 0, 2]], cr)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])


if __name__ == "__main__":
    unittest.main()
