"""
This module contains the function 'eval_x', used to evaluate the monthly
time-of-use (ToU) energy consumption associated with a typical load profile.

Notes
-----

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import numpy as np
from bill2watt.utils.check_sizes import check_y, check_x, check_nd
from bill2watt.common.common import nh, nj, fs, dt, arera


def evaluate(y, nd):
    """
    Evaluate the time-of-use (ToU) monthly consumption associated with a
    typical load profile.

    Notes
    -----
    None.

    Parameters
    ----------
    y : ndarray
        Hourly load profiles in the day-types.
    nd : ndarray
        Number of days of each type in the month.

    Returns
    -------
    x : ndarray
        Monthly consumption divided into ToU tariffs.
    """

    # Check consistency of data
    y = check_y(y)
    nd = check_nd(nd)

    # Evaluate bill (x) from the typical load profile (y) considering the
    # number of days of each type in the month (nd)
    x = np.array([sum([sum(y.reshape(nj, nh)[j, arera[j] == f]) * dt * nd[j]
                       for j in range(nj)]) for f in fs])
    return x
