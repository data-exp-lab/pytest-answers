import numpy as np
import pytest


answer_dir = "answers"


@pytest.mark.answer_test
def test_ones():
    return np.ones((10,)) * 2


@pytest.mark.answer_test
def test_dict():
    return {"a": 1, "b": 2}


@pytest.mark.answer_test
def test_str():
    return "somelongstring"


@pytest.mark.answer_test
def test_bytes():
    return b"somelongstring"


@pytest.mark.answer_test
def test_list_of_arrays():
    return [np.ones((2,)), np.ones((2,)) * 2]
