"""
This module contains various utility functions used throughout the package.

Functions
---------
eval_x(y, nd):
    Evaluate the time-of-use (ToU) monthly consumption associated with a
    typical load profile.

spread_y(y, nd):
    Generate a synthetic monthly load profile based on typical load profiles
    and weights of each profile.

Notes
-----

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import numpy as np
from bill2watt.utils.check_sizes import check_y, check_nd
from bill2watt.common.common import nh, nj, fs, dt, arera


def eval_x(y, nd):
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

    Raises
    ------
    AssertionError
        If the shapes of the input arrays do not meet the requirements.
        See 'chech_y' and 'check_nd' functions for further details.
    """

    # Check consistency of data
    y = check_y(y)
    nd = check_nd(nd)

    # Evaluate bill (x) from the typical load profile (y) considering the
    # number of days of each type in the month (nd)
    x = np.array([sum([sum(y.reshape(nj, nh)[j, arera[j] == f]) * dt * nd[j]
                       for j in range(nj)]) for f in fs])
    return x


def spread_y(y, nd):
    """
    Generate a synthetic monthly load profile based on typical load profile
    and weights of each profile, i.e., number of days of each day-type.

    Parameters
    ----------
    y : ndarray
        Typical load profiles representing different day types.
    nd : ndarray
        Array containing the number of days for each day type.

    Returns
    -------
    ndarray
        Synthetic monthly load profile.

    nd, axis=0)
    return synthetic_profile.flatten()

    Raises
    ------
    AssertionError
        If the shapes of the input arrays do not meet the requirements.
        See 'chech_y' and 'check_nd' functions for further details.
    """

    # Check consistency of data
    y = check_y(y)
    nd = check_nd(nd)

    # Spread the typical load profile in each day by the number of days
    y_spread = np.repeat(y.reshape(nj, nh), nd, axis=0)

    return y_spread.flatten()
