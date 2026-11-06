"""geckopy.__version__ is the installed version of the package."""
from importlib.metadata import version

import geckopy


def test_version_is_the_installed_version():
    assert geckopy.__version__ == version("geckopy")
