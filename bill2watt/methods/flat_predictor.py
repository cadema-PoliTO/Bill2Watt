"""
Flat Predictor

Description
-----------
This module contains the `FlatPredictor` class, which is used for predicting
typical load profiles based on monthly time-of-use (ToU) energy consumption
values, using a flat approach that spreads the consumption in each ToU tariff.

Notes
-----
None.

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import numpy as np
from bill2watt.methods.base_predictor import BasePredictor
from bill2watt.common.common import arera, fs


class FlatPredictor(BasePredictor):
    """
    Flat Predictor.

    This class implements a predictor for evaluating typical load profiles
    based on monthly time-of-use (ToU) energy consumption values using a flat
    approach that spreads the consumption in each ToU tariff.

    Parameters
    ----------

    Attributes
    ----------
    x_data : None
        Not used in this predictor.
    y_data : None
        Not used in this predictor.

    Methods
    -------
    predict(x, nd)
        Evaluate typical load profiles based on energy consumption values and
        number of days of each day-type.
    """

    def __init__(self):
        super().__init__(x_data=None, y_data=None)

    def predict(self, x, nd):
        """
        Evaluate typical load profiles in multiple points, based on ToU energy
        consumption values, using a flat approach.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values of the point(s) to predict.
        nd : array-like
            Number of days of each day-type for each point.

        Returns
        -------
        np.ndarray
            Predicted typical load profiles based on the provided inputs.

        Notes
        -----
        If 'x' (and other input parameters) refer to a single point, then
        the output is provided with a shape of a single point.

        Raises
        ------
        AssertionError
            If any inconsistency is found in the input data.
        """
        # Check consistency of input data
        x, nd, n_points, mono_dim = self.check_input_data(x, nd)

        # Initialize list of "predicted" typical load profiles
        y_pred = []

        # "Predict" typical load profile of each point
        for x_, nd_ in zip(x, nd):
            # Count hours of each tariff time-slot in each day-type
            n_hours = np.array([np.count_nonzero(arera == f, axis=1)
                                for f in fs])
            # Count hours of each tariff time-slot in the month
            n_hours = np.sum(nd_ * n_hours, axis=1)
            # Evaluate demand (flat) in each tariff time-slot
            k = x_ / n_hours
            # Evaluate load profile in each day-type assigning to each
            # time-step the related demand, according to ARERA's profiles
            y_pred_ = np.zeros_like(arera, dtype=float)
            for if_, f in enumerate(fs):
                y_pred_[arera == f] = k[if_]

            y_pred.append(y_pred_.flatten())

        if mono_dim:
            return np.array(y_pred)[0]
        else:
            return np.array(y_pred)

    def _fit(self):
        """
        Dummy method that raises an error indicating that fitting is not
         required for this predictor.

        Raises
        ------
        NotImplementedError
            Fitting is not required for the Flat Predictor.
        """
        raise NotImplementedError("Fitting is not required.")

    def add_data_points(self, x_data=None, y_data=None, **kwargs):
        """
        Add a data point to the predictor. This method is not supported by
        FlatPredictor and will raise an error unless 'x_data' and 'y_data'
        are both None.

        Parameters
        ----------
        x : None
            Energy consumption values of the data point.
        y : None
            Corresponding load profile of the data point.
        *+kwargs : dict
            Additional parameters, added for compatibility.

        Raises
        ------
        NotImplementedError
            This method is not supported by FlatPredictor and cannot be used
            unless 'x_data' and 'y_data' are both None.

        """
        if x_data is not None or y_data is not None:
            raise NotImplementedError("Adding data points is not supported.")


# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example usage
    flat_predictor = FlatPredictor()

    # Example usage with one point
    x = np.array([1, 2, 3])
    nd = np.array([22, 4, 5])
    y = flat_predictor.predict(x, nd)
    print("1 point evaluated, output shape: ", y.shape)
    plt.plot(y)
    plt.show()

    # Example usage with two points and scaling
    x = np.array([[1, 2, 3], [4, 5, 6]])
    nd = np.array([[22, 4, 5], [22, 4, 5]])
    y = flat_predictor.predict(x, nd)
    print("2 points evaluated, output shape: ", y.shape)
    plt.plot(y.T)
    plt.show()
