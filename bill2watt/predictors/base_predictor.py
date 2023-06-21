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
    x_data : pandas.DataFrame or None, optional
        ToU energy consumption values associated with the SLPs.
    y_data : pandas.DataFrame or None, optional
        Typical load profiles of the Standard Load Profiles (SLPs).

    Attributes
    ----------
    x_data (property) : pd.DataFrame or None
        ToU energy consumption values of the data points used to build the
        predictor. None for predictors that do not require data points.
    y_data (property) : pd.DataFrame or None
        Typical load profiles of the data points used to build the predictor.
        None for predictors that do not require data points.
    n (property) : int
        Number of data points used to build the predictor.

    Methods
    -------
    add_data(x_data, y_data, update_existing=True)
        Add new data points to the predictor.
    predict(x, nd)
        Placeholder method that must be implemented by child classes.
    """

    def __init__(self, x_data=None, y_data=None):
        self._x_data = None
        self._y_data = None
        self.add_data(x_data, y_data)

    @property
    def x_data(self):
        """
        Getter for the x_data property.

        Returns a copy of the x_data if it is not None, otherwise returns None.

        Returns
        -------
        pd.DataFrame or None
            Copy of the x_data or None.
        """
        return None if self._x_data is None else self._x_data.copy()

    @property
    def y_data(self):
        """
        Getter for the y_data property.

        Returns a copy of the y_data if it is not None, otherwise returns None.

        Returns
        -------
        pd.DataFrame or None
            Copy of the y_data or None.
        """
        return None if self._y_data is None else self._y_data.copy()

    @property
    def n(self):
        """
        Getter for the n property.

        Returns the number of data points (length of x_data or y_data).
        Returns None if both x_data and y_data are None.

        Returns
        -------
        int
            Number of data points.
        """
        if self.x_data is not None:
            return len(self.x_data)
        if self.y_data is not None:
            return len(self.y_data)
        return

    def __bool__(self):
        """
        Return True if data points have been added to the model.

        Parameters
        ----------

        Returns
        -------
        bool
            Whether data points have been added to the model ore not.
        """
        return self.n is not None

    def add_data(self, x_data=None, y_data=None, update_existing=True):
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
        x_data, y_data = self._validate_data(x_data, y_data)

        # Add new x and y data
        self._x_data = self._update_data(self.x_data, x_data,
                                         update_existing=update_existing)
        self._y_data = self._update_data(self.y_data, y_data,
                                         update_existing=update_existing)

        return self.x_data, self.y_data

    def _validate_data(self, x_data, y_data):
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

        def validate(data, number_columns):
            if data is not None:
                assert isinstance(data, pd.DataFrame), \
                    "Provided data must be None or a pandas.DataFrame object."
                assert len(data.columns) == number_columns, \
                    "Data must have {} columns.".format(number_columns)

        def validate_lengths_indices(x_data, y_data):
            if x_data is not None and y_data is not None:
                assert len(x_data) == len(y_data), \
                    "Lengths of 'x_data' and 'y_data' must match."
                assert x_data.index.equals(y_data.index), \
                    "Indices of 'x_data' and 'y_data' must match."

        # Validate input data
        validate(x_data, nf)
        validate(y_data, ni)
        validate_lengths_indices(x_data, y_data)

        # Validate w.r.t. to existing data
        if self.x_data is not None and self.y_data is not None:
            assert type(x_data) == type(y_data), \
                "The type of 'x_data' and 'y_data' must match."
        elif self.x_data is None and self.y_data is not None:
            assert x_data is None, "'x_data' must be None, only Y are used."
        elif self.y_data is None and self.x_data is not None:
            assert y_data is None, "'y_data' must be None, only X are used."

        return x_data, y_data

    def _fit(self):
        raise NotImplementedError("fit() method not implemented.")

    def predict(self, x, nd):
        raise NotImplementedError(
            "predict() method must be implemented by child classes.")

    @staticmethod
    def _update_data(data, new_data, update_existing=True):
        """
        Update existing data with new data.

        Parameters
        ----------
        data : pd.DataFrame or None
            Existing data to be updated.
        new_data : pd.DataFrame or None
            New data to update the existing data.
        update_existing : bool, optional
            Flag indicating whether to update existing data points (i.e.,
            already in the index). Default is True.

        Returns
        -------
        pd.DataFrame or None
            The updated data after merging existing and new data.
        """
        if new_data is None:
            return data
        if data is None:
            return new_data.copy()

        # Identify overlapping indices and store them separately
        overlap_indices = new_data.index.intersection(data.index)
        overlap_data = new_data.loc[overlap_indices].copy()
        new_data = new_data.drop(overlap_data.index)

        # Add new points, updating existing indexes if the flag is True
        if update_existing:
            data.loc[overlap_indices] = overlap_data
        return pd.concat((data, new_data))

        return data

    @staticmethod
    def _validate_input(x):
        """
        Check consistency of the input data for prediction.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values.

        Returns
        -------
        x : ndarray
            Updated energy consumption values.
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

        # Transform 'x' to array if not already
        x = np.array(x) if not isinstance(x, np.ndarray) else x

        # In case just one point is provided, "extend" quantities
        if x.ndim == 1:
            x = x[np.newaxis, :]
            mono_dim = True
            # Else, assert that the same number of points is provided
        elif x.ndim == 2:
            mono_dim = False
        else:
            raise ValueError("'x' must have one or two axes.")

        # Assert other dimensions
        assert x.shape[1] == nf, \
            "'x' must have size {} along axis 1.".format(nf)

        n_points = len(x)

        return x, n_points, mono_dim


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

    predictor.add_data(new_x_data, new_y_data,
                              update_existing=update_existing)
    print("Number of points after adding: ", len(predictor.x_data))
