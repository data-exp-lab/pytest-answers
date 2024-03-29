import os
import hashlib

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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--answers"):
        return
    skip_answer_test = pytest.mark.skip(reason="need --answers option to run")
    for item in items:
        if "answer_test" in item.keywords:
            item.add_marker(skip_answer_test)

def generate_test_name(item):
    """
    Generate a unique name for the hash for this test.
    """
    subdir = item.module.__name__.replace(".", "/")
    if item.cls is not None:
        fname = f"{item.cls.__name__}.{item.name}"
    else:
        fname = f"{item.name}"
    return os.path.join(subdir, fname)


def wrap_data_interceptor(plugin, item):
    if item.get_closest_marker("answer_test") is not None:
        test_name = generate_test_name(item)

        def answer_interceptor(store, obj):
            def wrapper(*args, **kwargs):
                store.return_value[test_name] = obj(*args, **kwargs)
            return wrapper
        item.obj = answer_interceptor(plugin, item.obj)


class AnswerComparison:
    def __init__(self, config, results_dir=None, store_dir=None):
        self.config = config
        self.store_dir = store_dir
        self.results_dir = results_dir

        # if not self.results_dir:
        #    self.results_dir = Path(tempfile.mkdtemp(dir=self.results_dir))
        self._generated_hash_library = {}
        self.return_value = {}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        marker = item.get_closest_marker("answer_test")

        if marker is None:
            yield
            return

        # Run test and get the resulting array
        wrap_data_interceptor(self, item)
        yield
        test_name = generate_test_name(item)
        if test_name not in self.return_value:
            # Test function did not complete successfully
            return
        answer = self.return_value[test_name]

        summary = {
            "status": None,
            "status_msg": None,
        }

        if self.store_dir is not None:
            summary["status"] = "skipped"
            summary['status_msg'] = "Skipped test, since generating answer."
            self.store_answer(item, answer)
            pytest.skip(summary['status_msg'])

        else:  # self.store_dir is None:
            # compare shit
            msg = self.compare_answer_to_store(item, answer, self.results_dir)
            if msg is not None:
                pytest.fail(msg, pytrace=False)

        #if item.cls is not None:
        #    setattr(item.cls, item.function.__name__, item_function_wrapper)
        ##else:
        #    item.obj = item_function_wrapper

    def pytest_unconfigure(self, config):
        """
        Save out the hash library at the end of the run.
        """
        # if self.generate_hash_library is not None:
        print("Do something here")

    def get_baseline_answer(self, item, result_dir):
        filename = generate_test_name(item) + ".h5"
        fullpath = os.path.join(result_dir, filename)
        if not os.path.exists(fullpath):
            pytest.fail(f"Answer '{filename}' does not exist", pytrace=False)
        with h5py.File(fullpath, "r") as f:
            if len(f.keys()) > 1:
                return {k: f[k][()] for k in f.keys()}
            else:
                return f["data"][()]

    def compare_answer_to_store(self, item, result, result_dir):
        reference = self.get_baseline_answer(item, result_dir)
        try:
            if isinstance(result, dict):  # assume dict of np.arrays
                for key in result:
                    assert key in reference
                    np.testing.assert_array_equal(reference[key], result[key])
            elif isinstance(result, bytes):
                assert reference == result  # TODO: be smarter about it
            elif isinstance(result, str):  # Strings are stored as bytes
                assert reference.decode() == result
            else:
                np.testing.assert_array_equal(reference, result)
        except AssertionError as exc:
            return str(exc)

    def store_answer(self, item, answer):
        # marker = item.get_closest_marker("answer_test")
        # get some extra options from marker if necessary

        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir, exist_ok=True)

        filename = generate_test_name(item) + ".h5"
        fullpath = os.path.join(self.store_dir, filename)
        if not os.path.isdir(os.path.dirname(fullpath)):
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)

        with h5py.File(fullpath, "a") as f:
            if isinstance(answer, dict):
                for k, v in answer.items():
                    ds = f.create_dataset(k, data=v)
                    if isinstance(v, np.ndarray):
                        ds.attrs["hash"] = hashlib.md5(v.tobytes()).hexdigest()
            else:
                ds = f.create_dataset("data", data=answer)
                if isinstance(answer, np.ndarray):
                    ds.attrs["hash"] = hashlib.md5(answer.tobytes()).hexdigest()
