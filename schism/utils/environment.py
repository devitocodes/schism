"""Objects for handling environment variables"""

import os
import numpy as np
from devito.logger import warning


def get_geometry_feps():
    """
    Amount of error expected in geometry handling. Larger values allow more
    slack in results of floating point calculations.
    """
    try:
        eps = float(os.environ['SCHISM_GEOM_FEPS'])
    except KeyError:
        eps = np.finfo(np.float32).eps
    except ValueError:
        warning("Geometry FEPS cannot be parsed, reverting to default")
        eps = np.finfo(np.float32).eps
    return eps
