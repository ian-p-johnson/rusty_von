import os

import pytest
import von


@pytest.fixture(autouse=True)
def reset_to_default_backend():
    yield
    # When the suite is pointed at a remote server (VON_TEST_BASE_URL), the
    # in-process engine is not the system under test: skip the reset so tests
    # never construct a local engine just to tear it down.
    if os.environ.get("VON_TEST_BASE_URL"):
        return
    von.set_backend("von-1.1")
