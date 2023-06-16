"""
This module contains the function 'evaluate', used to scale a typical load
profile according to certain ToU monthly energy consumption values, using a
flat approach.

Notes
-----

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import numpy as np
# from bill2watt.common.common import *
from bill2watt.utils import eval_x, check_y, check_x, check_nd


def evaluate(y, x_des, nd=None, x=None):
    """
    Scale a typical load profile (y) so that the total monthly energy
    consumption associated with it equals the desired value (x_des).

    Parameters
    ----------
    y : ndarray
        Typical load profile to be scaled.
    x_des : float or ndarray
        Desired monthly energy consumption (divided into ToU tariffs if ndarray).
    nd : ndarray, optional
        Number of days of each type in the month. If provided, 'x' should not
        be provided.
    x : float or ndarray, optional
        Monthly energy consumption associated with 'y' (divided into ToU
        tariffs if ndarray). If provided, 'nd' should not be provided.

    Returns
    -------
    y_scale : ndarray
        Scaled load profile.

    Raises
    ------
    AssertionError
        If 'y', 'x_des', 'nd', or 'x' do not comply with the
        requirements, or if 'nd' are 'x' are provided at the same time.
    """

    # Check consistency of data
    y = check_y(y)
    if isinstance(x_des, np.ndarray):
        x_des = check_x(x_des)
    else:
        x_des = check_x(np.array(x_des))
    assert (nd is None) ^ (x is None),\
        "Either 'nd' or 'x' should be provided, not both."
    if nd is not None:
        nd = check_nd(nd)
    if x is not None:
        if isinstance(x, np.ndarray):
            x = check_x(x)
        else:
            x = check_x(np.array(x))

    # Scale reference profiles

    # Evaluate ToU-monthly consumption associated with y, if needed
    if x is None:
        x = eval_x.evaluate(y, nd)
    # Calculate scaling factor
    k_scale = x_des.sum() / x.sum()
    # Evaluate scaled load profiles
    y_scale = y * k_scale

    return y_scale
