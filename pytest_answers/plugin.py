import os
import sys
import inspect
import tempfile
from pathlib import Path
from functools import wraps

import h5py
import numpy as np
import pytest


def pytest_addoption(parser):
    group = parser.getgroup("answer test comparison")
    group.addoption(
        "--answers",
        action="store_true",
        help="Enable comparison of answers to reference files",
    )
    group.addoption(
        "--answers-store",
        action="store",
        help="directory to generate reference answers in, relative to the location "
        " where py.test is run",
    )
    group.addoption(
        "--answers-dir",
        action="store",
        help="directory where reference answers are located, relative to the location "
        " where py.test is run",
    )


def pytest_configure(config):
    if config.getoption("--answers"):
        config.addinivalue_line(
            "markers",
            "answer_test: Compares resulting array against a previously stored version",
        )
        store_dir = config.getoption("--answers-store")
        results_dir = config.getoption("--answers-dir")
        config.pluginmanager.register(
            AnswerComparison(config, results_dir=results_dir, store_dir=store_dir)
        )


class AnswerComparison:
    def __init__(self, config, results_dir=None, store_dir=None):
        self.config = config
        self.store_dir = store_dir
        self.results_dir = results_dir

        # if not self.results_dir:
        #    self.results_dir = Path(tempfile.mkdtemp(dir=self.results_dir))
        self._generated_hash_library = {}

    def pytest_runtest_setup(self, item):
        marker = item.get_closest_marker("answer_test")

        if marker is None:
            return

        original = item.function

        @wraps(item.function)
        def item_function_wrapper(*args, **kwargs):
            if inspect.ismethod(original):
                result = original.__func__(*args, **kwargs)
            else:
                result = original(*args, **kwargs)

            # Generate answers
            if self.store_dir is not None:
                self.store_answer(item, result)

            msg = self.compare_answer_to_store(item, result, self.results_dir)

            if msg is not None:
                sys.stderr.write("ala\n")
                pytest.fail(msg, pytrace=False)

        if item.cls is not None:
            setattr(item.cls, item.function.__name__, item_function_wrapper)
        else:
            item.obj = item_function_wrapper

    def pytest_unconfigure(self, config):
        """
        Save out the hash library at the end of the run.
        """
        # if self.generate_hash_library is not None:
        print("Do something here")

    def generate_test_name(self, item):
        """
        Generate a unique name for the hash for this test.
        """
        return f"{item.module.__name__}.{item.name}"

    def get_baseline_answer(self, item, result_dir):
        filename = self.generate_test_name(item) + ".h5"
        with h5py.File(os.path.join(result_dir, filename), "r") as f:
            return f["data"][:]

    def compare_answer_to_store(self, item, result, result_dir):
        reference = self.get_baseline_answer(item, result_dir)
        try:
            np.testing.assert_array_equal(reference, result)
        except AssertionError as exc:
            return str(exc)

    def store_answer(self, item, answer):
        marker = item.get_closest_marker("answer_test")
        # get some extra options from marker if necessary

        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)  # TODO: make it thread-safe

        filename = self.generate_test_name(item) + ".h5"
        with h5py.File(os.path.join(self.store_dir, filename), "a") as f:
            f.create_dataset("data", data=answer)
