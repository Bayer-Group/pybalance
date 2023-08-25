from pybalance.utils import *
import pybalance.genetic
import pybalance.propensity
import pybalance.sim
import pybalance.visualization
import pybalance.lp

__version__ = "0.1.0"

import logging

logger = logging.getLogger(__name__)
logger.info(f"Loaded pybalance version {__version__}.")

# Logging is configured at the application level. To adjust logging level,
# configure logging as below before importing pybalance:
#
# import logging
# logging.basicConfig(
#     format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
#     level='INFO',
# )
#
