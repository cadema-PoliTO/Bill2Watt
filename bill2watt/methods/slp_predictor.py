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
from bill2watt.methods.base_predictor import BasePredictor

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
    x_data : pandas.DataFrame or None
        ToU energy consumption values associated with the SLPs.
    y_data : pandas.DataFrame
        Typical load profiles of the Standard Load Profiles (SLPs).

    Methods
    -------
    predict(x, key, nd=None, scaler=None)
        Evaluate typical load profiles based on ToU energy consumption values
        and using the SLPs.
    add_data_points(x, y)
        Add new data points to the predictor.
    """

    def __init__(self, x_data=None, y_data=None):
        if y_data is None:
            y_data = def_y_slp
            x_data = def_x_slp

        super().__init__(x_data, y_data)

    def predict(self, x, key, nd=None, scaler=None):
        """
        Evaluate typical load profiles in multiple points, based on ToU energy
        consumption values, using the SLPs.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values of the point(s) to predict.
        nd : array-like or None, optional
            Number of days of each day-type for each point.
            Only used if 'scaler' is not None. Default is None.
        key : tuple or list of tuples
            Key values related to the SLP associated with each point.
        scaler : callable or None, optional
            Scaling function to be applied to the predicted load profiles.
            Default is None, meaning that no scaling is performed.

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
        x, nd, n_points, mono_dim = self.check_input_data(x, nd)

        # If scaling is required, nd cannot be None as needed for scaling
        if scaler is not None:
            assert nd is not None, \
                "'nd' cannot be None, since needed for scaling."

        # Transform 'nd' to list of None for compatibility with zip
        nd = [None] * n_points if nd is None else nd

        # Check that right number of correct keys is provided
        key = (key,) if n_points == 1 and mono_dim else key
        assert len(key) == n_points, \
            "Length of 'key' must be equal to {}.".format(n_points)

        # Initialize list of "predicted" typical load profiles
        y_pred = []

        # "Predict" typical load profile of each point using SLPs
        for x_, nd_, key_ in zip(x, nd, key):
            assert key_ in self.y_data.index, \
                "Key {} is not present in the SLP index.".format(key_)

            y_pred_ = self.y_data.loc[key_, :].values

            # Scale, if necessary
            if scaler is not None:
                y_pred_ = scaler(y=y_pred_, x_des=x_, nd=nd_)

            y_pred.append(y_pred_)

        if mono_dim:
            return np.array(y_pred)[0]
        else:
            return np.array(y_pred)

    def _fit(self):
        """
        Dummy method indicating that fitting is not required.

        Raises
        ------
        NotImplementedError
            Always raised to indicate that fitting is not needed.
        """
        raise NotImplementedError("Fitting is not required. Use the "
                                  "'add_data_point' method to add SLPs.")


# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from bill2watt.scaling import flat

    # Example usage
    slp_predictor = SLPPredictor()

    # Example usage with one point
    x = np.array([1, 2, 3])
    nd = np.array([22, 4, 5])
    key = ('bta', 1)
    y = slp_predictor.predict(x, key, nd)
    print("1 point evaluated, output shape: ", y.shape)
    plt.plot(y)
    plt.show()

    # Example usage with two points and scaling
    x = np.array([[1, 2, 3], [4, 5, 6]])
    nd = np.array([[22, 4, 5], [22, 4, 5]])
    key = (('bta', 1), ('bta', 2))
    y = slp_predictor.predict(x, key,  nd, scaler=flat.evaluate)
    print("2 points evaluated, output shape: ", y.shape)
    plt.plot(y.T)
    plt.show()
