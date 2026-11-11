"""Functions that build the enzyme-data side of an ec model.

Mirrors GECKO MATLAB's `src/get_enzyme_data/` directory.
"""
from .copy_ec_to_gem import copy_ec_to_gem
from .ec_from_database import fill_eccodes_from_database, get_ec_from_database
from .ec_from_gem import fill_eccodes_from_gem, get_ec_from_gem
from .find_ec_in_db import find_ec_in_db

__all__ = [
    "copy_ec_to_gem",
    "fill_eccodes_from_database",
    "fill_eccodes_from_gem",
    "find_ec_in_db",
    "get_ec_from_database",  # deprecated; alias of fill_eccodes_from_database
    "get_ec_from_gem",  # deprecated; alias of fill_eccodes_from_gem
]
