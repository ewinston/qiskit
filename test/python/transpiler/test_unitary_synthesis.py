# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for the UnitarySynthesis transpiler pass.
"""

import unittest

from ddt import ddt, data

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeVigo
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.exceptions import TranspilerError


@ddt
class TestUnitarySynthesis(QiskitTestCase):
    """Test UnitarySynthesis pass."""

    def test_empty_basis_gates(self):
        """Verify when basis_gates is None, we do not synthesize unitaries."""
        qc = QuantumCircuit(1)
        qc.unitary([[0, 1], [1, 0]], [0])

        dag = circuit_to_dag(qc)

        out = UnitarySynthesis(None).run(dag)

        self.assertEqual(out.count_ops(), {"unitary": 1})

    @data(
        ["u3", "cx"],
        ["u1", "u2", "u3", "cx"],
        ["rx", "ry", "rxx"],
        ["rx", "rz", "iswap"],
        ["u3", "rx", "rz", "cz", "iswap"],
    )
    def test_two_qubit_synthesis_to_basis(self, basis_gates):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell_op = Operator(bell)

        qc = QuantumCircuit(2)
        qc.unitary(bell_op, [0, 1])
        dag = circuit_to_dag(qc)

        out = UnitarySynthesis(basis_gates).run(dag)

        self.assertTrue(set(out.count_ops()).issubset(basis_gates))

    def test_two_qubit_synthesis_to_directional_cx_from_gate_errors(self):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = FakeVigo()
        conf = backend.configuration()
        qr = QuantumRegister(2)
        coupling_map = CouplingMap(conf.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=False,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        unisynth_pass_nat = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )

        pm_nat = PassManager([triv_layout_pass, unisynth_pass_nat])
        qc_out_nat = pm_nat.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))
        self.assertEqual(Operator(qc), Operator(qc_out_nat))

    def test_swap_synthesis_to_directional_cx(self):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = FakeVigo()
        conf = backend.configuration()
        qr = QuantumRegister(2)
        coupling_map = CouplingMap(conf.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.swap(qr[0], qr[1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=False,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        unisynth_pass_nat = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )

        pm_nat = PassManager([triv_layout_pass, unisynth_pass_nat])
        qc_out_nat = pm_nat.run(qc)

        self.assertEqual(Operator(qc), Operator(qc_out))
        self.assertEqual(Operator(qc), Operator(qc_out_nat))

    def test_two_qubit_synthesis_to_directional_cx_multiple_registers(self):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = FakeVigo()
        conf = backend.configuration()
        qr0 = QuantumRegister(1)
        qr1 = QuantumRegister(1)
        coupling_map = CouplingMap(conf.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr0, qr1)
        qc.unitary(random_unitary(4, seed=12), [qr0[0], qr1[0]])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=False,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        unisynth_pass_nat = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=None,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )

        pm_nat = PassManager([triv_layout_pass, unisynth_pass_nat])
        qc_out_nat = pm_nat.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))
        self.assertEqual(Operator(qc), Operator(qc_out_nat))

    def test_two_qubit_synthesis_to_directional_cx_from_coupling_map(self):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = FakeVigo()
        conf = backend.configuration()
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=False,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        unisynth_pass_nat = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )

        pm_nat = PassManager([triv_layout_pass, unisynth_pass_nat])
        qc_out_nat = pm_nat.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))
        self.assertEqual(Operator(qc), Operator(qc_out_nat))

    def test_two_qubit_synthesis_not_pulse_optimal(self):
        """Verify not attempting pulse optimal decomposition when pulse_optimize==False."""
        backend = FakeVigo()
        conf = backend.configuration()
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=False,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)
        if isinstance(qc_out, QuantumCircuit):
            num_ops = qc_out.count_ops()  # pylint: disable=no-member
        else:
            num_ops = qc_out[0].count_ops()
        self.assertIn("sx", num_ops)
        self.assertGreaterEqual(num_ops["sx"], 16)

    def test_two_qubit_pulse_optimal_true_raises(self):
        """Verify not attempting pulse optimal decomposition when pulse_optimize==False."""
        from qiskit.exceptions import QiskitError

        backend = FakeVigo()
        conf = backend.configuration()
        # this assumes iswawp pulse optimal decomposition doesn't exist
        conf.basis_gates = [gate if gate != "cx" else "iswap" for gate in conf.basis_gates]
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        with self.assertRaises(QiskitError):
            pm.run(qc)

    def test_two_qubit_natural_direction_true_coupling_map_raises(self):
        """Verify not attempting pulse optimal decomposition when pulse_optimize==False."""
        # this assumes iswawp pulse optimal decomposition doesn't exist
        from qiskit.exceptions import QiskitError

        backend = FakeVigo()
        conf = backend.configuration()
        conf.basis_gates = [gate if gate != "cx" else "iswap" for gate in conf.basis_gates]
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 0], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        with self.assertRaises(QiskitError):
            pm.run(qc)

    def test_two_qubit_natural_direction_true_gate_length_raises(self):
        """Verify not attempting pulse optimal decomposition when pulse_optimize==False."""
        # this assumes iswawp pulse optimal decomposition doesn't exist
        backend = FakeVigo()
        conf = backend.configuration()
        for _, nduv in backend.properties()._gates["cx"].items():
            nduv["gate_length"] = (4e-7, nduv["gate_length"][1])
            nduv["gate_error"] = (7e-3, nduv["gate_error"][1])
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 0], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            backend_props=backend.properties(),
            pulse_optimize=True,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        with self.assertRaises(TranspilerError):
            pm.run(qc)

    def test_two_qubit_pulse_optimal_none_optimal(self):
        """Verify pulse optimal decomposition when pulse_optimize==None."""
        # this assumes iswawp pulse optimal decomposition doesn't exist
        backend = FakeVigo()
        conf = backend.configuration()
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=None,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)
        if isinstance(qc_out, QuantumCircuit):
            num_ops = qc_out.count_ops()  # pylint: disable=no-member
        else:
            num_ops = qc_out[0].count_ops()
        self.assertIn("sx", num_ops)
        self.assertLessEqual(num_ops["sx"], 12)

    def test_two_qubit_pulse_optimal_none_no_raise(self):
        """Verify pulse optimal decomposition when pulse_optimize==None."""
        # this assumes iswawp pulse optimal decomposition doesn't exist
        backend = FakeVigo()
        conf = backend.configuration()
        conf.basis_gates = [gate if gate != "cx" else "iswap" for gate in conf.basis_gates]
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=conf.basis_gates,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
            pulse_optimize=None,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)
        if isinstance(qc_out, QuantumCircuit):
            num_ops = qc_out.count_ops()  # pylint: disable=no-member
        else:
            num_ops = qc_out[0].count_ops()
        self.assertIn("sx", num_ops)
        self.assertLessEqual(num_ops["sx"], 14)


if __name__ == "__main__":
    unittest.main()
