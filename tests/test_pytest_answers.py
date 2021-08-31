import numpy as np
import pytest


answer_dir = "answers"


@pytest.mark.answer_test
def test_ones():
    return np.ones((10,)) * 2
