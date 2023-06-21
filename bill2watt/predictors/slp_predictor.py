"""
This module contains the 'SLPPredictor' class, which is used for predicting
typical load profiles based on monthly time-of-use (ToU) energy consumption
values, using Standard Load Profiles (SPLs).

Notes
-----

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import numpy as np
import pandas as pd
from os import path
from bill2watt.predictors.base_predictor import BasePredictor

# Default SLPs

# Path for the folder of common data
basepath = path.dirname(path.abspath(__file__))
folder = path.join(basepath, "data")

# Default Load Profiles (SLP) from GSE and associated energy bills
def_y_slp = pd.read_csv(path.join(folder, "y_slp_gse.csv"), sep=';',
                        index_col=[0, 1])
def_x_slp = pd.read_csv(path.join(folder, "x_slp_gse.csv"), sep=';',
                        index_col=[0, 1])


class SLPPredictor(BasePredictor):
    """
    This class implements a predictor for evaluating typical load profiles
    based on monthly time-of-use (ToU) energy consumption values, using
    Standard Load Profiles (SLPs).

    Parameters
    ----------
    x_data : pandas.DataFrame or None
        ToU energy consumption values associated with the SLPs.
        Since they are not used, can be also None.
    y_data : pandas.DataFrame or None
        Typical load profiles of the Standard Load Profiles (SLPs).
        If None, default values are used.

    Attributes
    ----------
    x_data (property) : pandas.DataFrame or None
        ToU energy consumption values of the training points.
        If None, default values are used.
    y_data (property) : pandas.DataFrame or None
        Typical load profiles of the training points.
        If None, default values are used.

    Methods
    -------
    predict(x, key, nd=None, scaler=None)
        Evaluate typical load profiles based on ToU energy consumption values
        and using the SLPs.
    add_data(x, y)
        Add new data points to the predictor.

    See Also
    --------
    BasePredictor : Base class for typical load profile predictors.
    """

    def __init__(self, x_data=None, y_data=None):
        if y_data is None:
            y_data = def_y_slp
            x_data = def_x_slp

        super().__init__(x_data, y_data)

    def predict(self, x, key):
        """
        Evaluate typical load profiles in multiple points, based on ToU energy
        consumption values, using the SLPs.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values of the point(s) to predict.
        key : tuple or list of tuples
            Key values related to the SLP associated with each point.

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
            If any inconsistency is found in the input data or key values are
            not present in the model data.
        """
        # Check consistency of input data
        x, n_points, mono_dim = self._validate_input(x)

        # Check that right number of correct keys is provided
        key = (key,) if n_points == 1 and mono_dim else key
        assert len(key) == n_points, \
            "Length of 'key' must be equal to {}.".format(n_points)

        # Initialize list of "predicted" typical load profiles
        y_pred = []

        # "Predict" typical load profile of each point using SLPs
        for x_, key_ in zip(x, key):
            assert key_ in self.y_data.index, \
                "Key {} is not present in the SLP index.".format(key_)

            y_pred.append(self.y_data.loc[key_, :].values)

        if mono_dim:
            return np.array(y_pred)[0]
        else:
            return np.array(y_pred)

    def _fit(self):
        """
        Dummy method indicating that fitting is not required.
        """
        print("Fitting is not required. Use 'add_data' method to add SLPs.")


# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from bill2watt.scaling import flat

    # Example usage
    predictor = SLPPredictor()

    # Example usage with one point
    x = np.array([1, 2, 3])
    nd = np.array([22, 4, 5])
    key = ('bta', 1)
    y = predictor.predict(x, key)
    print("1 point evaluated, output shape: ", y.shape)
    plt.plot(y)
    plt.show()

    # Example usage with two points and scaling
    x = np.array([[1, 2, 3], [4, 5, 6]])
    nd = np.array([[22, 4, 5], [22, 4, 5]])
    key = (('bta', 1), ('bta', 2))
    y = predictor.predict(x, key)
    print("2 points evaluated, output shape: ", y.shape)
    plt.plot(y.T)
    plt.show()
