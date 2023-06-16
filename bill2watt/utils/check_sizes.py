"""
This module contains functions for checking the size of various arrays.

Functions
---------
check_y(y)
    Check the size of y ndarray and return y.

check_x(x)
    Check the size of x ndarray and return x.

check_nd(nd)
    Check the size of nd ndarray and return nd.

Notes
-----

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito
"""

from bill2watt.common.common import ni, nf, nj
import numpy as np


def check_y(y):
    """
    Check the size of y ndarray.

    Parameters
    ----------
    y : ndarray
        Typical load profile to be checked.

    Returns
    -------
    ndarray
        The input 'y'.

    Raises
    ------
    AssertionError
        If the size of 'y' does not comply with the requirements.
    """
    if __debug__:
        assert np.size(y) == ni,\
            "'y' must have a size equal to {}.".format(ni)
    return y


def check_x(x):
    """
    Check the size of x ndarray.

    Parameters
    ----------
    x : ndarray
        Energy consumption values to be checked.

    Returns
    -------
    ndarray
        The input 'x'.

    Raises
    ------
    AssertionError
        If the size of 'x' does not comply with the requirements.
    """
    if __debug__:
        assert np.size(x) == nf,\
            "'x' must have size equal to {}.".format(nf)
    return x


def check_nd(nd):
    """
    Check the size of nd ndarray.

    Parameters
    ----------
    nd : ndarray
        Number of days of each type to be checked.

    Returns
    -------
    ndarray
        The input 'nd'.

    Raises
    ------
    AssertionError
        If the size of 'nd' does not comply with the requirements.
    """
    if __debug__:
        assert np.size(nd) == nj,\
            "'nd' must have size equal to {}.".format(nj)
    return nd
