"""Test to verify testing framework setup."""

import pytest
from hypothesis import given, strategies as st


def test_pytest_works():
    """Verify pytest is working."""
    assert True


@given(st.integers())
def test_hypothesis_works(x):
    """Verify hypothesis is working."""
    assert isinstance(x, int)
