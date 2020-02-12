# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Controlled unitary gate.
"""

from typing import List, Optional, Union
from qiskit.circuit.exceptions import CircuitError
from .gate import Gate


class ControlledGate(Gate):
    """Controlled unitary gate.
    
    Attributes:
        num_ctrl_qubits: The number of control qubits to add.
        base_gate: An instance of the target unitary to control.
    """

    def __init__(self, name: str, num_qubits: int, params: List,
                 label: Optional[str] = None, num_ctrl_qubits: Optional[int] = 1,
                 definition: Optional[List] = None):
        """Create a new ControlledGate. In the new gate the first ``num_ctrl_qubits``
        of the gate are the controls.

        Args:
            name: The name of the gate.
            num_qubits: The number of qubits the gate acts on.
            params: A list of parameters for the gate.
            label: An optional label for the gate.
            num_ctrl_qubits: Number of control qubits.
            definition: A list of gate rules for implementing this gate. The
                elements of the list are tuples of (:meth:`~qiskit.circuit.Gate`, [qubit_list],
                [clbit_list]).

        Raises:
            CircuitError: If ``num_ctrl_qubits`` >= ``num_qubits``.

        Examples:

        Create a controlled standard gate and apply it to a circuit.

        .. jupyter-execute::

           from qiskit import QuantumCircuit, QuantumRegister
           from qiskit.extensions.standard import HGate

           qr = QuantumRegister(3)
           qc = QuantumCircuit(qr)
           c3h_gate = HGate().control(2)
           qc.append(c3h_gate, qr)
           qc.draw()

        Create a controlled custom gate and apply it to a circuit.

        .. jupyter-execute::

           from qiskit import QuantumCircuit, QuantumRegister
           from qiskit.extensions.standard import HGate

           qc1 = QuantumCircuit(2)
           qc1.x(0)
           qc1.h(1)
           custom = qc1.to_gate().control(2)

           qc2 = QuantumCircuit(4)
           qc2.append(custom, [0, 3, 1, 2])
           qc2.draw()
        """
        super().__init__(name, num_qubits, params, label=label)
        if num_ctrl_qubits < num_qubits:
            self.num_ctrl_qubits = num_ctrl_qubits
        else:
            raise CircuitError('number of control qubits must be less than the number of qubits')
        if definition:
            self.definition = definition
            if len(definition) == 1:
                base_gate = definition[0][0]
                if isinstance(base_gate, ControlledGate):
                    self.base_gate = base_gate.base_gate
                else:
                    self.base_gate = base_gate

    def __eq__(self, other) -> bool:
        if not isinstance(other, ControlledGate):
            return False
        else:
            return (other.num_ctrl_qubits == self.num_ctrl_qubits and
                    self.base_gate == other.base_gate and
                    super().__eq__(other))

    def inverse(self) -> 'ControlledGate':
        """Invert this gate by calling inverse on the base gate."""
        return self.base_gate.inverse().control(self.num_ctrl_qubits)
