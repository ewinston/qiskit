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

"""Cancel the redundant (self-adjoint) gates through commutation relations."""

import math
from collections import deque
import numpy as np
import pandas as pd
from qiskit.quantum_info import Operator
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.axis_angle_analysis import (AxisAngleAnalysis,
                                                                       _su2_axis_angle)
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Gate


_CUTOFF_PRECISION = 1E-5


class AxisAngleReduction(TransformationPass):
    """Reduce runs of single qubit gates with common axes.
    """

    def __init__(self, basis_gates=None):
        """
        AxisAngleReduction initializer.
        """
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()

        self.requires.append(AxisAngleAnalysis())


    def run(self, dag):
        """Run the AxisAngleReduction pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        self._commutation_analysis()
        dfprop = self.property_set['axis-angle']
        del_list = list()
        for wire in dag.wires:
            node_it = dag.nodes_on_wire(wire)
            stack = list()  # list of (node, dfprop index)
            for node in node_it:
                if node.type != 'op' or not isinstance(node.op, Gate):
                    del_list += self._eval_stack(stack, dag)
                    stack = list()
                    continue
                # just doing 1q for now
                if len(node.qargs) != 1:
                    del_list += self._eval_stack(stack, dag)
                    stack = list()
                    continue
                if not stack:
                    stack.append((node, self._get_index(node._node_id)))
                    continue
                top_node = stack[-1][0]
                top_index = self._get_index(top_node._node_id)
                this_node = node
                this_index = self._get_index(this_node._node_id)
                top_group = dfprop.iloc[top_index].basis_group
                this_group = dfprop.iloc[this_index].basis_group
                if top_group == this_group:
                    stack.append((this_node, this_index))
                elif len(stack) > 1:
                    del_list += self._eval_stack(stack, dag)
                    stack = [(this_node, this_index)]  # start new stack with this valid op
            del_list += self._eval_stack(stack, dag)
        for node in del_list:
            dag.remove_op_node(node)
        return dag

    def _eval_stack(self, stack, dag):
        dfprop = self.property_set['axis-angle']
        if len(stack) <= 1:
            return []
        top_node = stack[-1][0]
        top_index = self._get_index(top_node._node_id)
        var_gate = dfprop.iloc[top_index].var_gate
        if var_gate and not self._symmetry_complete(stack):
            del_list = self._reduce_stack(stack, var_gate, dag)
        else:
            del_list = self._symmetry_cancellation(stack, dag)
        return del_list

    def _get_index(self, idnode):
        """return the index in dfprop where idop occurs"""
        dfprop = self.property_set['axis-angle']
        return dfprop.index[dfprop.id == idnode][0]

    def _symmetry_cancellation(self, stack, dag):
        """Elliminate gates by symmetry. This doesn't require a
        variable rotation gate for the axis.
        Args:
            stack (list(DAGNode, int)): All nodes share a rotation axis and the int
                indexes the node in the dataframe.
            dag (DAGCircuit): the whole dag. Will not be modified.

        Returns:
            list(DAGNode): List of dag nodes to delete from dag ultimately.
        """
        if len(stack) <= 1:
            return []
        dfprop = self.property_set['axis-angle']
        stack_nodes, stack_indices = zip(*stack)
        del_list = []
        del_list_stack_indices = []
        # get contiguous symmetry groups
        dfsubset = dfprop.iloc[list(stack_indices)]
        symmetry_groups = dfsubset.groupby(
            (dfsubset.symmetry_order.shift() != dfsubset.symmetry_order).cumsum())
        for _, dfsym in symmetry_groups:
            sym_order = dfsym.iloc[0].symmetry_order
            if sym_order == 1:
                # no rotational symmetry
                continue
            num_cancellation_groups, _ = divmod(len(dfsym), sym_order)
            groups_phase = dfsym.phase.iloc[0:num_cancellation_groups * sym_order].sum()
            if num_cancellation_groups == 0:
                # not enough members to satisfy symmetry cancellation
                continue
            if num_cancellation_groups % 2:  # double cover (todo:improve conditionals)
                dag.global_phase += np.pi
            if math.cos(groups_phase) == -1:
                dag.global_phase += np.pi
            del_ids = dfsym.iloc[0:num_cancellation_groups * sym_order].id
            this_del_list = [dag.node(delId) for delId in del_ids]
            del_list += this_del_list
            # get indices of nodes in stack and remove from stack
            del_list_stack_indices += [stack_nodes.index(node)
                                       for node in this_del_list]
        red_stack = [nodepair for inode, nodepair in enumerate(stack)
                     if inode not in del_list_stack_indices]
        if len(red_stack) < len(stack):
            # stack modified; attempt further cancellation recursively
            del_list += self._symmetry_cancellation(red_stack, dag)
        return del_list

    def _reduce_stack(self, stack, var_gate_name, dag):
        """reduce common axis rotations to single rotation. This requires
        a single parameter rotation gate for the axis. Multiple parameter would
        be possible (e.g. RGate, RVGate) if one had a generic way of identifying how to rotate
        by the specified angle if the angle is not explicitly denoted in the gate arguments."""
        if not stack:
            return []
        dfprop = self.property_set['axis-angle']
        _, stack_indices = zip(*stack)
        smask = list(stack_indices)
        del_list = []
        dfsubset = dfprop.iloc[smask]
        dfsubset['var_gate_angle'] = dfsubset.angle * dfsubset.rotation_sense
        params = dfsubset[['var_gate_angle', 'phase']].sum()
        if np.mod(params.var_gate_angle, (2 * np.pi)) > _CUTOFF_PRECISION:
            var_gate = self.property_set['var_gate_class'][var_gate_name](params.var_gate_angle)
            new_qarg = QuantumRegister(1, 'q')
            new_dag = DAGCircuit()
            # the variable gate for the axis may not be in this stack
            df_gate = dfprop[dfprop.name == var_gate_name]
            df_gate_phase = df_gate.phase
            df_gate_angle = df_gate.angle
            df_gate_phase_factor = df_gate_phase / df_gate_angle
            phase_factor_uni = df_gate_phase_factor.unique()
            if len(phase_factor_uni) == 1 and np.isfinite(phase_factor_uni[0]):
                gate_phase_factor = phase_factor_uni[0]
            else:
                _, _, gate_phase_factor = _su2_axis_angle(Operator(var_gate).data)
            new_dag.global_phase = params.phase - params.var_gate_angle * gate_phase_factor
            new_dag.add_qreg(new_qarg)
            new_dag.apply_operation_back(var_gate, [new_qarg[0]])
            dag.substitute_node_with_dag(stack[0][0], new_dag)
            del_list += [node for node, _ in stack[1:]]
        else:
            del_list += [node for node, _ in stack]
        return del_list

    def _commutation_analysis(self, rel_tol=1e-9, abs_tol=0.0):
        dfprop = self.property_set['axis-angle']
        buniq = dfprop.axis.unique()  # basis unique
        # merge collinear axes iff either contains a variable rotation
        naxes = len(buniq)
        # index pairs of buniq which are vectors in opposite directions
        buniq_inverses = list()
        buniq_parallel = list()
        vdot = np.full((naxes, naxes), np.nan)
        for v1_ind in range(naxes):
            v1 = buniq[v1_ind]
            for v2_ind in range(v1_ind + 1, naxes):
                v2 = buniq[v2_ind]
                vdot[v1_ind, v2_ind] = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        buniq_parallel = list(zip(*np.where(np.isclose(vdot, 1, rtol=rel_tol, atol=abs_tol))))
        buniq_inverses = list(zip(*np.where(np.isclose(vdot, -1, rtol=rel_tol, atol=abs_tol))))
        buniq_common = buniq_parallel + buniq_inverses
        grouped_common = [list(group) for group in join_if_intersect(buniq_common)]

        dfprop['basis_group'] = None
        # "rotation sense" is used to indicate sense of rotation wrt : +1=ccw, -1=cw
        dfprop['rotation_sense'] = 1
        # name of variable rotation gate for the basis group if it exists, else None
        dfprop['var_gate'] = None
        # count the number of independent bases
        basis_counter = 0
        unlabeled_axes = list(range(naxes))
        # determine if inverses have arbitrary single parameter rotation
        mask_1p = dfprop.nparams == 1
        for group in grouped_common:
            lead = group[0]  # this will be the reference direction for the group
            mask = pd.Series(False, index=range(dfprop.shape[0]))
            for member in group:
                mask |= dfprop.axis == buniq[member]
                unlabeled_axes.remove(member)
            dfprop.loc[mask, 'basis_group'] = basis_counter
            if (dfprop[mask].nparams == 1).any():
                group_single_param_name = dfprop[mask & mask_1p].name
                if group_single_param_name.any():
                    var_gate_name = dfprop[mask & mask_1p].name.iloc[0]
                    dfprop.loc[mask, 'var_gate'] = var_gate_name
            basis_counter += 1
            # create mask for inverses to lead
            mask[:] = False
            for pair in buniq_inverses:
                try:
                    mask |= dfprop.axis == buniq[pair[int(not pair.index(lead))]]
                except ValueError:
                    # positive sense lead is not in pair; skip
                    pass
            dfprop.loc[mask, 'rotation_sense'] = -1
        # index lone bases
        for bindex in unlabeled_axes[:]:
            mask = dfprop.axis == buniq[bindex]
            dfprop.loc[mask, 'basis_group'] = basis_counter
            if (dfprop[mask].nparams == 1).any():
                var_gate_name = dfprop.loc[mask & mask_1p].name.iloc[0]
                dfprop.loc[mask, 'var_gate'] = var_gate_name
            unlabeled_axes.remove(bindex)
            basis_counter += 1

    def _symmetry_complete(self, stack):
        """Determine whether complete cancellation is possible due to symmetry"""
        dfprop = self.property_set['axis-angle']
        _, stack_indices = zip(*stack)
        sym_order = dfprop.iloc[list(stack_indices)].symmetry_order
        sym_order_zero = sym_order.iloc[0]
        return (sym_order_zero == len(sym_order)) and all(sym_order_zero == sym_order)


def join_if_intersect(lists):
    """This is from user 'agf' on stackoverflow
    https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    """
    results = []
    if not lists:
        return results
    sets = deque(set(lst) for lst in lists if lst)
    disjoint = 0
    current = sets.pop()
    while True:
        merged = False
        newsets = deque()
        for _ in range(disjoint, len(sets)):
            this = sets.pop()
            if not current.isdisjoint(this):
                current.update(this)
                merged = True
                disjoint = 0
            else:
                newsets.append(this)
                disjoint += 1
        if sets:
            newsets.extendleft(sets)
        if not merged:
            results.append(current)
            try:
                current = newsets.pop()
            except IndexError:
                break
            disjoint = 0
        sets = newsets
    return results
