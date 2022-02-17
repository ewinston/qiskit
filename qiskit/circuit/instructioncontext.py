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
This class represents an element of the 'data' element of a quantum circuit. In particular
it is meant to contain contextual information about the the following elmeents:

   Instruction: the type of operation performed. If a gate, the unbound parameterized form.
   cargs: classical bits of the instruction
   qargs: quantum bits of the instruction
   scalar_params: the scalar params of the instruction used to determine the matrix form of the
      of the instruction if it is a Gate as well as possibly it's decomposition, which may depend
      on the numeric representation.

Here we try the standard library's dataclass. From PEP #557 some other options are

    - collections.namedtuple in the standard library.
    - typing.NamedTuple in the standard library.
    - The popular attrs [1] project.
    - George Sakkis' recordType recipe [2], a mutable data type inspired by collections.namedtuple.
    - Many example online recipes [3], packages [4], and questions [5]. David Beazley used a form of 
      data classes as the motivating example in a PyCon 2013 metaclass talk [6].
"""
from dataclasses import dataclass, astuple
from typing import (
    List,
    Tuple,
    Union
)
import qiskit
from qiskit.circuit.instruction import Instruction

@dataclass
class InstructionContext:
    """
    Class for storing Instruction and it's circuit context which help specify 
    how Instruction behaves in cthe QuantumCircuit. It is an element of the QuantumCircuit.data
    list.
    """
    instruction: Instruction
    qargs: Tuple['QubitSpecifier']
    cargs: Tuple['ClbitSpecifier']
    params: Tuple[Union[float,'ParameterExpression']] = tuple()


    def __array__(self):
        if isinstance(self, Gate):
            return self.to_matrix(*self.params)

    def __iter__(self):
        return iter(astuple(self))

    def inverse(self):
        """
        Returns a new instance like the QuantumCircuit method of the same name.
        """
        return InstructionContext(self.instruction.inverse(), self.qargs, self.cargs, self.params)

    def reverse(self):
        print("TODO: implement")
