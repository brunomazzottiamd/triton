# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------


# pytest
import pytest


# Quick test fixture.
# ------------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--quick-test", action="store_true", help="enable quick test mode")


@pytest.fixture
def quick_test(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--quick-test")
