#!/usr/bin/env python3

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

# Based heavily on subunit2junit filter script from the upstream subunit python
# package, https://github.com/testing-cabal/subunit/ just adapted to run on
# python3 more reliably.

"""Script to convert subunit stream returned by stestr to junitxml for processing by azure-pipelines."""

import argparse
import sys

from junitxml import JUnitXmlResult  # pylint: disable=import-error
from subunit.filters import run_tests_from_stream
from testtools import StreamToExtendedDecorator


def filter_by_result(
    result_factory,
    output_path,
    passthrough,
    forward,
    input_stream=sys.stdin,
    protocol_version=1,
    passthrough_subunit=True,
):
    """Filter an input stream using a test result.

    :param callable result_factory: A callable that when passed an output stream
        returns a TestResult.  It is expected that this result will output
        to the given stream.
    :param str output_path: A path send output to.  If None, output will be go
        to ``sys.stdout``.
    :param bool passthrough: If True, all non-subunit input will be sent to
        ``sys.stdout``.  If False, that input will be discarded.
    :param bool forward: If True, all subunit input will be forwarded directly to
        ``sys.stdout`` as well as to the ``TestResult``.
    :param file input_stream: The source of subunit input.  Defaults to
        ``sys.stdin``.
    :param int protocol_version: The subunit protocol version to expect.
    :param bool passthrough_subunit: If True, passthrough should be as subunit.

    :return: A test result with the results of the run.
    :rtype: JUnitXmlResult
    """
    if protocol_version == 1:
        sys.stderr.write("Subunit protocol version 2 must be used")
        sys.exit(1)

    if passthrough:
        passthrough_stream = sys.stdout
    else:
        passthrough_stream = None

    if forward:
        forward_stream = sys.stdout
    else:
        forward_stream = None

    if output_path is None:
        output_to = sys.stdout
    else:
        output_to = open(output_path, "w")

    try:
        result = result_factory(output_to)
        run_tests_from_stream(
            input_stream,
            result,
            passthrough_stream,
            forward_stream,
            protocol_version=protocol_version,
            passthrough_subunit=passthrough_subunit,
        )
    finally:
        if output_path:
            output_to.close()
    return result


def run_filter_script(
    result_factory, description, post_run_hook=None, protocol_version=1, passthrough_subunit=True
):
    """Main function for simple subunit filter scripts.

    Many subunit filter scripts take a stream of subunit input and use a
    TestResult to handle the events generated by that stream.  This function
    wraps a lot of the boiler-plate around that by making a script with
    options for handling passthrough information and stream forwarding

    :param callable result_factory: A callable that takes an output stream and returns
        a test result that outputs to that stream.
    :param str description: A description of the filter script.
    :param callable post_run_hook: A callback function that runs after the test run
        finishes. It will be passed a single positional argument the result
        object returned by the run.
    :param int protocol_version: What protocol version to consume/emit.
    :param bool passthrough_subunit: If True, passthrough should be as subunit.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--no-passthrough",
        action="store_true",
        help="Hide all non subunit input.",
        default=False,
        dest="no_passthrough",
    )
    parser.add_argument("-o", "--output-to", help="Send the output to this path rather than stdout")
    parser.add_argument(
        "-f",
        "--forward",
        action="store_true",
        default=False,
        help="Forward subunit stream on stdout. When set, "
        "received non-subunit output will be encapsulated"
        " in subunit.",
    )
    args = parser.parse_args()
    result = filter_by_result(
        result_factory,
        args.output_to,
        not args.no_passthrough,
        args.forward,
        protocol_version=protocol_version,
        passthrough_subunit=passthrough_subunit,
        input_stream=sys.stdin,
    )
    if post_run_hook:
        post_run_hook(result)
    if not hasattr(result, "wasSuccessful"):
        result = result.decorated


def _main():
    run_filter_script(
        lambda output: StreamToExtendedDecorator(JUnitXmlResult(output)),
        "Convert to junitxml",
        protocol_version=2,
    )


if __name__ == "__main__":
    _main()
