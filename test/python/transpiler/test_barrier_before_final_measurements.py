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

"""Test the BarrierBeforeFinalMeasurements pass"""

import unittest
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit.circuit import QuantumRegister, QuantumCircuit, ClassicalRegister, Clbit
from qiskit.test import QiskitTestCase


class TestBarrierBeforeFinalMeasurements(QiskitTestCase):
    """Tests the BarrierBeforeFinalMeasurements pass."""

    def test_single_measure(self):
        """A single measurement at the end
                          |
        q:--[m]--     q:--|-[m]---
             |    ->      |  |
        c:---.---     c:-----.---
        """
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_ignore_single_measure(self):
        """Ignore single measurement because it is not at the end
        q:--[m]-[H]-      q:--[m]-[H]-
             |        ->       |
        c:---.------      c:---.------
        """
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.h(qr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.h(qr[0])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_single_measure_mix(self):
        """Two measurements, but only one is at the end
                                                |
        q0:--[m]--[H]--[m]--     q0:--[m]--[H]--|-[m]---
              |         |    ->        |        |  |
         c:---.---------.---      c:---.-----------.---
        """
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.h(qr)
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.h(qr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_two_qregs(self):
        """Two measurements in different qregs to different cregs
                                          |
        q0:--[H]--[m]------     q0:--[H]--|--[m]------
                   |                      |   |
        q1:--------|--[m]--  -> q1:-------|---|--[m]--
                   |   |                  |   |   |
        c0:--------.---|---      c0:----------.---|---
                       |                          |
        c1:------------.---      c0:--------------.---
        """
        qr0 = QuantumRegister(1, "q0")
        qr1 = QuantumRegister(1, "q1")
        cr0 = ClassicalRegister(1, "c0")
        cr1 = ClassicalRegister(1, "c1")

        circuit = QuantumCircuit(qr0, qr1, cr0, cr1)
        circuit.h(qr0)
        circuit.measure(qr0, cr0)
        circuit.measure(qr1, cr1)

        expected = QuantumCircuit(qr0, qr1, cr0, cr1)
        expected.h(qr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0)
        expected.measure(qr1, cr1)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_two_qregs_to_a_single_creg(self):
        """Two measurements in different qregs to the same creg
                                          |
        q0:--[H]--[m]------     q0:--[H]--|--[m]------
                   |                      |   |
        q1:--------|--[m]--  -> q1:-------|---|--[m]--
                   |   |                  |   |   |
        c0:--------.---|---     c0:-----------.---|---
           ------------.---        ---------------.---
        """
        qr0 = QuantumRegister(1, "q0")
        qr1 = QuantumRegister(1, "q1")
        cr0 = ClassicalRegister(2, "c0")

        circuit = QuantumCircuit(qr0, qr1, cr0)
        circuit.h(qr0)
        circuit.measure(qr0, cr0[0])
        circuit.measure(qr1, cr0[1])

        expected = QuantumCircuit(qr0, qr1, cr0)
        expected.h(qr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0[0])
        expected.measure(qr1, cr0[1])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_preserve_measure_for_conditional(self):
        """Test barrier is inserted after any measurements used for conditionals

        q0:--[H]--[m]------------     q0:--[H]--[m]-------|-------
                   |                             |        |
        q1:--------|--[ z]--[m]--  -> q1:--------|--[ z]--|--[m]--
                   |    |    |                   |    |       |
        c0:--------.--[=1]---|---     c0:--------.--[=1]------|---
                             |                                |
        c1:------------------.---     c1:---------------------.---
        """
        qr0 = QuantumRegister(1, "q0")
        qr1 = QuantumRegister(1, "q1")
        cr0 = ClassicalRegister(1, "c0")
        cr1 = ClassicalRegister(1, "c1")
        circuit = QuantumCircuit(qr0, qr1, cr0, cr1)

        circuit.h(qr0)
        circuit.measure(qr0, cr0)
        circuit.z(qr1).c_if(cr0, 1)
        circuit.measure(qr1, cr1)

        expected = QuantumCircuit(qr0, qr1, cr0, cr1)
        expected.h(qr0)
        expected.measure(qr0, cr0)
        expected.z(qr1).c_if(cr0, 1)
        expected.barrier(qr0, qr1)
        expected.measure(qr1, cr1)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))


class TestBarrierBeforeMeasurementsWhenABarrierIsAlreadyThere(QiskitTestCase):
    """Tests the BarrierBeforeFinalMeasurements pass when there is a barrier already"""

    def test_handle_redundancy(self):
        """The pass is idempotent
            |                |
        q:--|-[m]--      q:--|-[m]---
            |  |     ->      |  |
        c:-----.---      c:-----.---
        """
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_preserve_barriers_for_measurement_ordering(self):
        """If the circuit has a barrier to enforce a measurement order,
        preserve it in the output.

         q:---[m]--|-------     q:---|--[m]--|-------
           ----|---|--[m]--  ->   ---|---|---|--[m]--
               |       |                 |       |
         c:----.-------|---     c:-------.-------|---
           ------------.---       ---------------.---
        """
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr[0], cr[0])
        expected.barrier(qr)
        expected.measure(qr[1], cr[1])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_measures_followed_by_barriers_should_be_final(self):
        """If a measurement is followed only by a barrier,
        insert the barrier before it.

         q:---[H]--|--[m]--|-------     q:---[H]--|--[m]-|-------
           ---[H]--|---|---|--[m]--  ->   ---[H]--|---|--|--[m]--
                       |       |                      |      |
         c:------------.-------|---     c:------------.------|---
           --------------------.---       -------------------.---
        """
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])

        expected = QuantumCircuit(qr, cr)
        expected.h(qr)
        expected.barrier(qr)
        expected.measure(qr[0], cr[0])
        expected.barrier(qr)
        expected.measure(qr[1], cr[1])

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_should_merge_with_smaller_duplicate_barrier(self):
        """If an equivalent barrier exists covering a subset of the qubits
        covered by the new barrier, it should be replaced.

         q:---|--[m]-------------     q:---|--[m]-------------
           ---|---|---[m]--------  ->   ---|---|---[m]--------
           -------|----|---[m]---       ---|---|----|---[m]---
                  |    |    |                  |    |    |
         c:-------.----|----|----     c:-------.----|----|----
           ------------.----|----       ------------.----|----
           -----------------.----       -----------------.----
        """
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr[0], qr[1])
        circuit.measure(qr, cr)

        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_should_merge_with_larger_duplicate_barrier(self):
        """If a barrier exists and is stronger than the barrier to be inserted,
        preserve the existing barrier and do not insert a new barrier.

         q:---|--[m]--|-------     q:---|--[m]-|-------
           ---|---|---|--[m]--  ->   ---|---|--|--[m]--
           ---|---|---|---|---       ---|---|--|---|---
                  |       |                 |      |
         c:-------.-------|---     c:-------.------|---
           ---------------.---       --------------.---
           -------------------       ------------------
        """
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")

        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])

        expected = circuit

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barrier_doesnt_reorder_gates(self):
        """A barrier should not allow the reordering of gates, as pointed out in #2102

        q:--[p(0)]----------[m]---------      q:--[p(0)]-----------|--[m]---------
          --[p(1)]-----------|-[m]------  ->    --[p(1)]-----------|---|-[m]------
          --[p(2)]-|---------|--|-[m]----       --[p(2)]-|---------|---|--|-[m]----
          ---------|-[p(03)]-|--|--|-[m]-       ---------|-[p(03)]-|---|--|--|-[m]-
                             |  |  |  |                                |  |  |  |
        c:-------------------.--|--|--|-     c:------------------------.--|--|--|-
          ----------------------.--|--|-       ---------------------------.--|--|-
          -------------------------.--|-       ------------------------------.--|-
          ----------------------------.-       ---------------------------------.-

        """

        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circuit = QuantumCircuit(qr, cr)

        circuit.p(0, qr[0])
        circuit.p(1, qr[1])
        circuit.p(2, qr[2])
        circuit.barrier(qr[2], qr[3])
        circuit.p(3, qr[3])

        test_circuit = circuit.copy()
        test_circuit.measure(qr, cr)

        # expected circuit is the same, just with a barrier before the measurements
        expected = circuit.copy()
        expected.barrier(qr)
        expected.measure(qr, cr)

        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(test_circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_conditioned_on_single_bit(self):
        """Test that the pass can handle cases where there is a loose-bit condition."""
        circuit = QuantumCircuit(QuantumRegister(3), ClassicalRegister(2), [Clbit()])
        circuit.h(range(3))
        circuit.measure(range(3), range(3))
        circuit.h(0).c_if(circuit.cregs[0], 3)
        circuit.h(1).c_if(circuit.clbits[-1], True)
        circuit.h(2).c_if(circuit.clbits[-1], False)
        circuit.measure(range(3), range(3))

        expected = circuit.copy_empty_like()
        expected.h(range(3))
        expected.measure(range(3), range(3))
        expected.h(0).c_if(expected.cregs[0], 3)
        expected.h(1).c_if(expected.clbits[-1], True)
        expected.h(2).c_if(expected.clbits[-1], False)
        expected.barrier(range(3))
        expected.measure(range(3), range(3))

        pass_ = BarrierBeforeFinalMeasurements()
        self.assertEqual(expected, pass_(circuit))


class TestControlFlow(QiskitTestCase):
    """Tests the BarrierBeforeFinalMeasurements pass."""

    def test_simple_if_else(self):
        """Test that the pass is not confused by if-else."""
        pass_ = BarrierBeforeFinalMeasurements()

        base_test = QuantumCircuit(1, 1)
        base_test.z(0)
        base_test.measure(0, 0)

        test = QuantumCircuit(1, 1)
        test.if_else(
            (test.clbits[0], True), base_test.copy(), base_test.copy(), test.qubits, test.clbits
        )
        test.measure(0, 0)

        expected = QuantumCircuit(1, 1)
        expected.if_else(
            (expected.clbits[0], True),
            base_test.copy(),
            base_test.copy(),
            expected.qubits,
            expected.clbits,
        )
        expected.barrier(0)
        expected.measure(0, 0)
        test_pass = pass_(test)
        self.assertEqual(test_pass, expected)

    def test_final_measure_in_if_else(self):
        """Test if-else containing final measure."""
        pass_ = BarrierBeforeFinalMeasurements()

        base_test = QuantumCircuit(2, 1)
        base_test.z(0)
        base_test.measure(0, 0)

        qreg = QuantumRegister(2, "q")
        creg = ClassicalRegister(1)
        test = QuantumCircuit(qreg, creg)
        test.if_else((test.clbits[0], True), base_test.copy(), base_test.copy(), qreg[[0, 1]], creg)

        base_expected = QuantumCircuit(2, 1)
        base_expected.z(0)
        base_expected.barrier(base_expected.qubits)
        base_expected.measure(0, 0)
        expected = QuantumCircuit(qreg, creg)
        expected.if_else(
            (expected.clbits[0], True),
            base_expected.copy(),
            base_expected.copy(),
            expected.qubits,
            expected.clbits,
        )
        test_pass = pass_(test)
        self.assertEqual(test_pass, expected)

    def test_control_flow_not_final(self):
        """Test non-final control flow that internally has 'final' measurements"""
        pass_ = BarrierBeforeFinalMeasurements()

        base_test = QuantumCircuit(2, 1)
        base_test.z(0)
        base_test.measure(0, 0)

        qreg = QuantumRegister(2, "q")
        creg = ClassicalRegister(1)
        test = QuantumCircuit(qreg, creg)
        test.if_else((test.clbits[0], True), base_test.copy(), base_test.copy(), qreg[[0, 1]], creg)
        test.x(0)

        test_pass = pass_(test)
        self.assertEqual(test_pass, test)

    def test_nested_control_flow(self):
        """Test barrier in nested control flow."""
        pass_ = BarrierBeforeFinalMeasurements()

        test_level2 = QuantumCircuit(2, 1)
        test_level2.cz(0, 1)
        test_level2.measure(0, 0)

        test_level1 = QuantumCircuit(2, 1)
        test_level1.while_loop(
            (test_level1.clbits[0], True),
            test_level2.copy(),
            test_level1.qubits,
            test_level1.clbits,
        )
        test = QuantumCircuit(2, 1)
        test.for_loop((0,), None, test_level1.copy(), test.qubits, [])

        test_pass = pass_(test)

        body_expected = QuantumCircuit(2, 1)
        body_expected.for_loop((0,), None, test_level1.copy(), body_expected.qubits, [])
        body_expected.measure(0, 0)

        expected_level2 = QuantumCircuit(2, 1)
        expected_level2.cz(0, 1)
        expected_level2.barrier(expected_level2.qubits)
        expected_level2.measure(0, 0)

        expected_level1 = QuantumCircuit(2, 1)
        expected_level1.while_loop(
            (expected_level1.clbits[0], True),
            expected_level2,
            expected_level1.qubits,
            expected_level1.clbits,
        )

        expected = QuantumCircuit(2, 1)
        expected.for_loop((0,), None, expected_level1.copy(), expected.qubits, [])
        self.assertEqual(test_pass, expected)

    def test_control_flow_with_no_measure(self):
        """Test no barrier inserted if control flow has no final measure"""
        pass_ = BarrierBeforeFinalMeasurements()

        base_test = QuantumCircuit(1, 1)
        base_test.z(0)

        test = QuantumCircuit(1, 1)
        test.if_else(
            (test.clbits[0], True), base_test.copy(), base_test.copy(), test.qubits, test.clbits
        )
        test_pass = pass_(test)
        self.assertEqual(test_pass, test)


if __name__ == "__main__":
    unittest.main()
