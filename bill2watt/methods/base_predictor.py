"""
This module contains the 'BasePredictor' class, which is used as a base for the
models that predict typical load profiles based on monthly time-of-use (ToU)
energy consumption values.

Notes
-----

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import numpy as np
import pandas as pd
from bill2watt.common.common import nf, ni, nj


class BasePredictor:
    """
    Base class for predictors of typical load profiles (y) based on monthly
    time-of-use (ToU) energy consumption values (x).

    Parameters
    ----------
    x_data : pandas.DataFrame or None
        ToU energy consumption values associated with the SLPs.
    y_data : pandas.DataFrame or None
        Typical load profiles of the Standard Load Profiles (SLPs).

    Attributes
    ----------
    x_data : pd.DataFrame or None
        ToU energy consumption values of the data points used to build the
        predictor. None for predictors that do not require data points.
    y_data : pd.DataFrame or None
        Typical load profiles of the data points used to build the predictor.
        None for predictors that do not require data points.

    Methods
    -------
    add_data_points(x, y)
        Add new data points to the predictor.
    predict()
        Placeholder method, that must be implemented by child classes.
    """

    def __init__(self, x_data=None, y_data=None):
        self.x_data, self.y_data = None, None
        self.add_data_points(x_data, y_data)

    def _check_data_consistency(self, x_data, y_data):
        """
        Check consistency of the data points used to build the model.

        Parameters
        ----------
        x_data : pd.DataFrame or None
            ToU energy consumption values of the data points.
            None if no data points are provided.
        y_data : pd.DataFrame or None
            Typical load profiles of the data points.
            None if no data points are provided.

        Returns
        -------
        pd.DataFrame or None
            The input 'x_data'.
        pd.DataFrame or None
            The input 'y_data'.

        Raises
        ------
        AssertionError
            If 'x_data' and 'y_data' do not comply with the current status
            of self.x_data and self.y_data.
            If 'x_data' is not None and not a pandas DataFrame object.
            If 'y_data' is not None and not a pandas DataFrame object.
            If 'x_data' and 'y_data' are not None and their lengths or indices
            do not match, or their columns do not comply with the requirements.
        """

        # Checks depending on the status of self.x_data and self.y_data
        if self.x_data is not None and self.y_data is not None:
            assert type(x_data) == type(y_data),\
                "The type of 'x_data' and 'y_data' must match."
        elif self.x_data is None and self.y_data is not None:
            assert x_data is None, \
                "'x_data' must be None since only x_data are used."
        elif self.y_data is None and self.x_data is not None:
            assert y_data is None, \
                "'y_data' must be None since only y_data are used."

        # Basic checks on the provided x_data and y_data
        assert x_data is None or isinstance(x_data, pd.DataFrame), \
            "'x_data' must either be None or a pandas.DataFrame object."
        assert y_data is None or isinstance(y_data, pd.DataFrame), \
            "'y_data' must either be None or a pandas.DataFrame object."
        if x_data is not None and y_data is not None:
            assert len(x_data) == len(y_data), \
                "Lengths of 'x_data' and 'y_data' must match."
            assert x_data.index.equals(y_data.index), \
                "Indices of x_data and y_data must match."
        if x_data is not None:
            assert len(x_data.columns) == nf, \
                "'x_data' must have {} columns.".format(nf)
        if y_data is not None:
            assert len(y_data.columns) == ni, \
                "'y_data' must have {} columns.".format(ni)

        return x_data, y_data

    def add_data_points(self, x_data=None, y_data=None, update_existing=True):
        """
        Add new data points to the predictor.

        Parameters
        ----------
        x_data : pd.DataFrame or None
            ToU energy consumption values of the new data points.
            None if no data points are provided.
        y_data : pd.DataFrame or None
            Typical load profiles of the new data points.
            None if no data points are provided.
        update_existing : bool, optional
            Flag indicating whether to update existing data points (i.e.,
            already in the index). Default is True.

        Returns
        -------
        pd.DataFrame or None
            The updated x_data after adding new data points.
        pd.DataFrame or None
            The updated y_data after adding new data points.

        Raises
        ------
        AssertionError
            If 'x_data' and 'y_data' do not comply with requirements for new
            data points.
        """
        x_data, y_data = self._check_data_consistency(x_data, y_data)

        if self.x_data is not None:
            # Identify overlapping indices and store them separately
            overlap_indices = x_data.index.intersection(self.x_data.index)
            x_overlap = x_data.loc[overlap_indices].copy()
            x_data = x_data.drop(x_overlap.index)
            # Add new points, updating existing indexes if flag is True
            if update_existing:
                self.x_data.loc[overlap_indices] = x_overlap
            self.x_data = pd.concat((self.x_data, x_data))
        elif x_data is not None:
            self.x_data = x_data.copy()
        else:
            self.x_data = x_data

        if self.y_data is not None:
            # Identify overlapping indices and store them separately
            overlap_indices = y_data.index.intersection(self.y_data.index)
            y_overlap = y_data.loc[overlap_indices].copy()
            y_data = y_data.drop(y_overlap.index)
            # Add new points, updating existing indexes if flag is True
            if update_existing:
                self.y_data.loc[overlap_indices] = y_overlap
            self.y_data = pd.concat((self.y_data, y_data))
        elif y_data is not None:
            self.y_data = y_data.copy()
        else:
            self.y_data = y_data

        return self.x_data, self.y_data

    def _fit(self):
        raise NotImplementedError("fit() method not implemented.")

    def predict(self, x, nd):
        raise NotImplementedError("predict() method not implemented.")

    @staticmethod
    def check_input_data(x, nd=None):
        """
        Check consistency of the input data.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values.
        nd : array-like or None, optional
            Number of days of each day-type for each point. Default is None.

        Returns
        -------
        x : ndarray
            Updated energy consumption values.
        nd : ndarray or None
            Updated number of days.
        n_points : int
            Number of points.
        mono_dim : bool
            Whether the input data is mono-dimensional.

        Notes
        -----
        If 'x' (and other input parameters) refer to a single point, then
        they are added a dimension, while returning 'mono_dim' equal to True.

        Raises
        ------
        AssertionError
            If any inconsistency is found in the input data.
        """

        # Transform 'x' and 'nd' to arrays if not already
        x = np.array(x) if not isinstance(x, np.ndarray) else x
        if not isinstance(nd, np.ndarray) and nd is not None:
            nd = np.array(nd)

        # In case just one point is provided, "extend" quantities
        if x.ndim == 1 and (nd is None or nd.ndim == 1):
            x = x[np.newaxis, :]
            if nd is not None:
                nd = nd[np.newaxis, :]
            mono_dim = True
            # Else, assert that the same number of points is provided
        else:
            if nd is not None:
                assert len(x) == len(nd), "'x' and 'nd' must have same length."
            mono_dim = False

        # Assert other dimensions
        assert x.shape[1] == nf, \
            "'x' must have size {} along axis 1.".format(nf)
        if nd is not None:
            assert nd.shape[1] == nj, \
                "'nd' must have size {} along axis 1.".format(nj)

        n_points = len(x)

        return x, nd, n_points, mono_dim


# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import numpy.random as npr
    from bill2watt.common.common import nf, ni

    # Make a predictor and add new points
    n = 3  # points
    n_new = 2  # new_points
    new_x_index = range(n, n + n_new)  # try overlapping index
    new_y_index = new_x_index  # try non-matching indices
    update_existing = True

    x_data = pd.DataFrame(npr.rand(n, nf))  # try combinations of None
    y_data = pd.DataFrame(npr.rand(n, ni))  # try combinations of None

    new_x_data = pd.DataFrame(npr.rand(n_new, nf), index=new_x_index)
    new_y_data = pd.DataFrame(npr.rand(n_new, ni), index=new_y_index)

    predictor = BasePredictor(x_data, y_data)
    print("Number of points before adding: ", len(predictor.x_data))

    predictor.add_data_points(new_x_data, new_y_data,
                              update_existing=update_existing)
    print("Number of points after adding: ", len(predictor.x_data))
