"""General utility functions for ecModels.

Mirrors GECKO MATLAB's `src/geckomat/utilities/` directory.
"""
from .add_new_rxns_to_ec import (
    AddNewRxnsResult,
    NewEnzyme,
    add_new_rxns_to_ec,
)
from .bottlenecks import get_enzyme_bottlenecks
from .ec_fseof import EcFSEOFResult, ec_fseof
from .ec_fva import ec_fva
from .enzyme_usage import EnzymeUsageResult, enzyme_usage
from .get_subset_ec_model import get_subset_ec_model
from .load_conventional_gem import load_conventional_gem
from .load_ec_model import load_ec_model
from .map_rxns_to_conv import MapRxnsResult, map_rxns_to_conv
from .pfba_enzymes import pfba_enzymes
from .report_enzyme_usage import EnzymeUsageReport, report_enzyme_usage
from .save_ec_model import save_ec_model

__all__ = [
    "AddNewRxnsResult",
    "EcFSEOFResult",
    "EnzymeUsageReport",
    "EnzymeUsageResult",
    "MapRxnsResult",
    "NewEnzyme",
    "add_new_rxns_to_ec",
    "ec_fseof",
    "ec_fva",
    "enzyme_usage",
    "get_enzyme_bottlenecks",
    "get_subset_ec_model",
    "load_conventional_gem",
    "load_ec_model",
    "map_rxns_to_conv",
    "pfba_enzymes",
    "report_enzyme_usage",
    "save_ec_model",
]
