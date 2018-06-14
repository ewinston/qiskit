# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""IBMQJob module

This module is used for creating asynchronous job objects for the
IBM Q Experience.
"""

from concurrent import futures
import time
import logging
import pprint
import json
import numpy

from qiskit.backends import BaseJob
from qiskit.backends.jobstatus import JobStatus
from qiskit._qiskiterror import QISKitError
from qiskit._result import Result
from qiskit._resulterror import ResultError
from qiskit._compiler import compile_circuit

logger = logging.getLogger(__name__)


class IBMQJob(BaseJob):
    """IBM Q Job class """

    def __init__(self, q_job, api, is_device):
        """IBMQJob init function.

        Args:
            q_job (QuantumJob): job description
            api (IBMQuantumExperience): IBM Q API
            is_device (bool): whether backend is a real device  # TODO: remove this after Qobj
        """
        super().__init__()
        qobj = q_job.qobj
        self._job_data = {
            'circuits': qobj['circuits'],
            'hpc':  None if 'hpc' not in qobj['config'] else qobj['config']['hpc'],
            'seed': qobj['circuits'][0]['config']['seed'],  # TODO <-- [0] ???
            'shots': qobj['config']['shots'],
            'max_credits': qobj['config']['max_credits']
        }
        self._api = api
        self._job_id = None
        self._backend_name = qobj.get('config').get('backend_name')
        self._status = JobStatus.INITIALIZING
        self._status_msg = None
        self._cancelled = False
        self._is_device = is_device
        self._queue_position = 0
        self._creation_date = None
        self._executor = futures.ThreadPoolExecutor()
        self._future = None

    @classmethod
    def from_api(cls, job_info, api, is_device):
        """Instantiates job using information returned from
        IBMQuantumExperience about a particular job.

        Args:
            job_info (dict): This is the information about a job returned from
                the API. It has the simplified structure:

                {'backend': {'id', 'backend id string',
                             'name', 'ibmqx4'},
                 'id': 'job id string',
                 'qasms': [{'executionId': 'id string',
                            'qasm': 'qasm string'},
                          ]
                 'status': 'status string',
                 'seed': '1',
                 'shots': 1024,
                 'status': 'status string',
                 'usedCredits': 3,
                 'creationDate': '2018-06-13T04:31:13.175Z'
                 'userId': 'user id'}
            api (IBMQuantumExperience): IBM Q API
            is_device (bool): whether backend is a real device  # TODO: remove this after Qobj

        Returns:
            IBMQJob: an instance of this class
        """
        job_instance = cls.__new__(cls)
        job_instance._is_device = is_device
        job_instance._cancelled = False
        job_instance._status_msg = None
        job_instance._queue_position = 0
        job_instance._executor = futures.ThreadPoolExecutor()
        job_instance._future = None
        job_instance._backend_name = job_info.get('backend').get('name')
        job_instance._api = api
        job_instance._job_id = job_info.get('id')
        # update status (need _api and _job_id)
        # pylint: disable=pointless-statement
        job_instance._creation_date = job_info.get('creationDate')
        job_instance.status
        return job_instance

    # pylint: disable=arguments-differ
    def result(self, timeout=None, wait=5):
        """Return the result from the job.

        Args:
           timeout (int): number of seconds to wait for job
           wait (int): time between queries to IBM Q server

        Returns:
            Result: Result object

        Raises:
            IBMQJobError: exception raised during job initialization
        """
        self._wait_for_submitting()
        this_result = self._wait_for_job(timeout=timeout, wait=wait)
        if self._is_device:
            _reorder_bits(this_result)
        return this_result

    def cancel(self):
        """Attempt to cancel job.
           TODO: Currently this is only possible on commercial systems.
        Returns:
            bool: True if job can be cancelled, else False.

        Raises:
            IBMQJobError: if server returned error
        """
        self._wait_for_submitting()
        if self._is_commercial():
            hub = self._api.config['hub']
            group = self._api.config['group']
            project = self._api.config['project']
            response = self._api.cancel_job(self._job_id, hub, group, project)
            if 'error' in response:
                err_msg = response.get('error', '')
                raise IBMQJobError('Error cancelling job: %s' % err_msg)
            else:
                self._cancelled = True
                return True
        else:
            self._cancelled = False
            return False

    @property
    def status(self):
        """ Query the API for the status of the Job """
        self._wait_for_submitting()
        api_job = self._api.get_job(self._job_id)
        if 'status' not in api_job:
            raise IBMQJobError("get_job didn't return status: %s" %
                               (pprint.pformat(api_job)))
        elif api_job['status'] == 'RUNNING':
            self._status = JobStatus.RUNNING
            # we may have some other information here
            if 'infoQueue' in api_job:
                if 'status' in api_job['infoQueue']:
                    if api_job['infoQueue']['status'] == 'PENDING_IN_QUEUE':
                        self._status = JobStatus.QUEUED
                if 'position' in api_job['infoQueue']:
                    self._queue_position = api_job['infoQueue']['position']
        elif api_job['status'] == 'COMPLETED':
            self._status = JobStatus.DONE
        elif api_job['status'] == 'CANCELLED':
            self._cancelled = True
            self._status = JobStatus.CANCELLED
        elif 'ERROR' in api_job['status']:
            # ERROR_CREATING_JOB or ERROR_RUNNING_JOB
            self._status = JobStatus.ERROR
            self._status_msg = api_job['status']
        else:
            self._status = JobStatus.ERROR
            raise IBMQJobError('Unexpected behavior of {0}\n{1}'.format(
                self.__class__.__name__,
                pprint.pformat(api_job)))
        return self._status

    def queue_position(self):
        """
        Returns the position in the server queue

        Returns:
            Number: Position in the queue. 0 = No queued.
        """
        return self._queue_position

    @property
    def queued(self):
        """
        Returns whether job is queued.

        Returns:
            bool: True if job is queued, else False.

        Raises:
            QISKitError: couldn't get job status from server
        """
        return self.status == JobStatus.QUEUED

    @property
    def running(self):
        """
        Returns whether job is actively running

        Returns:
            bool: True if job is running, else False.

        Raises:
            IBMQJobError: couldn't get job status from server
        """
        return self.status == JobStatus.RUNNING

    @property
    def done(self):
        """
        Returns True if job successfully finished running.

        Note behavior is slightly different than Future objects which would
        also return true if successfully cancelled.
        """
        return self.status == JobStatus.DONE

    @property
    def cancelled(self):
        return self._cancelled

    @property
    def job_id(self):
        """
        Return backend determined job_id.
        """
        self._wait_for_submitting()
        return self._job_id

    @property
    def backend_name(self):
        """
        Return backend name used for this job
        """
        return self._backend_name

    def submit(self):
        """ Submits a Job to the concurrent pool executor """
        if self._future is not None:
            raise IBMQJobError("We have already submitted a job!")

        hpc = self._job_data['hpc']
        seed = self._job_data['seed']
        shots = self._job_data['shots']
        max_credits = self._job_data['max_credits']

        api_jobs = []
        if 'circuits' in self._job_data:
            circuits = self._job_data['circuits']
            for circuit in circuits:
                job = _create_job_from_circuit(circuit)
                api_jobs.append(job)

        hpc_camel_cased = None
        if hpc is not None:
            try:
                # Use CamelCase when passing the hpc parameters to the API.
                hpc_camel_cased = {
                    'multiShotOptimization': hpc['multi_shot_optimization'],
                    'ompNumThreads': hpc['omp_num_threads']
                }
            except (KeyError, TypeError):
                hpc_camel_cased = None

        self._future = self._executor.submit(self._submit_callback, api_jobs,
                                             self._backend_name, hpc_camel_cased,
                                             seed, shots, max_credits)

    def _submit_callback(self, api_jobs, backend_name, hpc, seed, shots,
                         max_credits):
        """Submit job to IBM Q.

        Parameters:
            api_jobs (list): List of API Job dictionaries to submit. One per circuit.
            backend_name (string): The name of the backend
            hpc: HPC specific configuration
            seed: The seed for the circuits
            shots: Number of shots the circuits should run
            max_credits: Maximum number of credits

        Raises:
            QISKitError: The backend name in the job doesn't match this backend.
            ResultError: If the API reported an error with the submitted job.
            RegisterSizeError: If the requested register size exceeded device
                capability.
        """
        return self._api.run_job(api_jobs, backend=backend_name,
                                 shots=shots, max_credits=max_credits,
                                 seed=seed, hpc=hpc)

    def _wait_for_job(self, timeout=60, wait=5):
        """Wait until all online ran circuits of a qobj are 'COMPLETED'.

        Args:
            timeout (float or None): seconds to wait for job. If None, wait
                indefinitely.
            wait (float): seconds between queries

        Returns:
            Result: A result object.

        Raises:
            IBMQJobError: job didn't return status or reported error in status
        """
        timer = 0
        job_result = JobResult(self._job_id, 'CANCELLED', '')

        while not self._cancelled:
            if timeout is not None and timer >= timeout:
                job_result = JobResult(self._job_id, 'ERROR', 'QISkit Time Out')
                break

            api_result = self._api.get_job(self._job_id)
            if 'status' not in api_result:
                raise IBMQJobError("get_job didn't return status: %s" %
                                   (pprint.pformat(api_result)))

            if 'ERROR' in api_result['status'].upper():
                logger.info('There was an error in the Job: status = %s', api_result['status'])
                self._status = JobStatus.ERROR
                job_result = JobResult(self._job_id, 'ERROR', api_result['status'])
                break

            if api_result['status'] == 'RUNNING':
                logger.info('status = %s (%d seconds)', api_result['status'], timer)
                time.sleep(wait)
                timer += wait
                continue

            if api_result['status'] == 'CANCELLED':
                self._status = JobStatus.CANCELLED
                job_result = JobResult(self._job_id, 'CANCELLED', api_result['status'])
                logger.info('The job was cancelled!')
                break

            # After this point the job is completed!
            logger.info('Job is complete! status = %s', api_result['status'])
            job_result_list = []
            for circuit_result in api_result['qasms']:
                this_result = {'data': circuit_result['data'],
                               'name': circuit_result.get('name'),
                               'compiled_circuit_qasm': circuit_result.get('qasm'),
                               'status': circuit_result['status']}
                if 'metadata' in circuit_result:
                    this_result['metadata'] = circuit_result['metadata']
                job_result_list.append(this_result)

            logger.info('Got a result from remote backend %s with job id: %s',
                        self.backend_name, self._job_id)
            job_result = JobResult(self._job_id, api_result['status'],
                                   job_result_list,
                                   api_result.get('usedCredits'),
                                   backend_name=self._backend_name)
            self._status = JobStatus.DONE
            break

        return Result(job_result.as_dict())

    def _is_commercial(self):
        config = self._api.config
        # this check may give false positives so should probably be improved
        return config.get('hub') and config.get('group') and config.get('project')

    def _wait_for_submitting(self):
        if self._job_id is None:
            if self._future is None:
                raise IBMQJobError("You have to submit before asking for status or results!")
            submit_info = self._future.result(timeout=60)
            if 'error' in submit_info:
                raise IBMQJobError(str(submit_info['error']))

            self._creation_date = submit_info.get('creationDate')
            self._status = JobStatus.QUEUED
            self._job_id = submit_info.get('id')


def _reorder_bits(result):
    """temporary fix for ibmq backends.
    for every ran circuit, get reordering information from qobj
    and apply reordering on result"""
    for circuit_result in result._result['result']:
        if 'metadata' not in circuit_result:
            raise QISKitError('result object missing metadata for reordering bits')

        circ = circuit_result['metadata'].get('compiled_circuit')
        # device_qubit -> device_clbit (how it should have been)
        measure_dict = {op['qubits'][0]: op['clbits'][0] for op in circ['operations']
                        if op['name'] == 'measure'}
        counts_dict_new = {}
        for item in circuit_result['data']['counts'].items():
            # fix clbit ordering to what it should have been
            bits = list(item[0])
            bits.reverse()  # lsb in 0th position
            count = item[1]
            reordered_bits = list('x' * len(bits))
            for device_clbit, bit in enumerate(bits):
                if device_clbit in measure_dict:
                    correct_device_clbit = measure_dict[device_clbit]
                    reordered_bits[correct_device_clbit] = bit
            reordered_bits.reverse()

            # only keep the clbits specified by circuit, not everything on device
            num_clbits = circ['header']['number_of_clbits']
            compact_key = reordered_bits[-num_clbits:]
            compact_key = "".join([b if b != 'x' else '0'
                                   for b in compact_key])

            # insert spaces to signify different classical registers
            cregs = circ['header']['clbit_labels']
            if sum([creg[1] for creg in cregs]) != num_clbits:
                raise ResultError("creg sizes don't add up in result header.")
            creg_begin_pos = []
            creg_end_pos = []
            acc = 0
            for creg in reversed(cregs):
                creg_size = creg[1]
                creg_begin_pos.append(acc)
                creg_end_pos.append(acc + creg_size)
                acc += creg_size
            compact_key = " ".join([compact_key[creg_begin_pos[i]:creg_end_pos[i]]
                                    for i in range(len(cregs))])

            # marginalize over unwanted measured qubits
            if compact_key not in counts_dict_new:
                counts_dict_new[compact_key] = count
            else:
                counts_dict_new[compact_key] += count

        circuit_result['data']['counts'] = counts_dict_new


def _numpy_type_converter(obj):
    if isinstance(obj, numpy.integer):
        return int(obj)
    elif isinstance(obj, numpy.floating):  # pylint: disable=no-member
        return float(obj)
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()
    return obj


def _create_job_from_circuit(circuit):
    """ Helper function that creates a special Job required by the API, from a circuit
    """
    job = {}
    if (('compiled_circuit_qasm' not in circuit) or
            (circuit['compiled_circuit_qasm'] is None)):
        compiled_circuit = compile_circuit(circuit['circuit'])
        circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)

    if isinstance(circuit['compiled_circuit_qasm'], bytes):
        job['qasm'] = circuit['compiled_circuit_qasm'].decode()
    else:
        job['qasm'] = circuit['compiled_circuit_qasm']

    if 'name' in circuit:
        job['name'] = circuit['name']

    # convert numpy types for json serialization
    compiled_circuit = json.loads(json.dumps(circuit['compiled_circuit'],
                                             default=_numpy_type_converter))
    job['metadata'] = {'compiled_circuit': compiled_circuit}
    return job


class IBMQJobError(QISKitError):
    """class for IBM Q Job errors"""
    pass


class JobResult:
    """ Just a Helper class to encapsulate the results taken from a Job"""
    def __init__(self, job_id, status, result, used_circuits=None,
                 backend_name=None):
        self._job_id = job_id
        self._status = status
        self._result = result
        self._used_circuits = used_circuits
        self._backend_name = backend_name

    def as_dict(self):
        """ Transforms into a dictionary """
        ret = {
            'job_id': self._job_id,
            'status': self._status,
            'result': self._result,
        }
        if self._used_circuits is not None:
            ret['used_circuit'] = self._used_circuits
        if self._backend_name is not None:
            ret['backend_name'] = self._backend_name

        return ret
