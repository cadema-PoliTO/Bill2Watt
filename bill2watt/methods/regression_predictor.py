"""
This module contains the 'RegressionPredictor' class, which is used for
predicting typical load profiles based on monthly time-of-use (ToU) energy
consumption values, using training data and in a regression approach.

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
from sklearn.neighbors import KNeighborsRegressor
from bill2watt.methods.base_predictor import BasePredictor

# Path for the folder of common data
basepath = path.dirname(path.abspath(__file__))
folder = path.join(basepath, "data")

# Default training dataset
def_y_data = pd.read_csv(path.join(folder, "y_train.csv"), sep=';',
                         index_col=[0, 1])
def_x_data = pd.read_csv(path.join(folder, "x_train.csv"), sep=';',
                         index_col=[0, 1])

# Default regressor
default_regressor = KNeighborsRegressor(n_neighbors=9)


class RegressionPredictor(BasePredictor):
    """
    This class implements a predictor for evaluating typical load profiles
    based on time-of-use (ToU) energy consumption values using training
    data in a regression approach.

    Parameters
    ----------
    x_data : pandas.DataFrame or None
        ToU energy consumption values of the training points.
        Since they are not used, can be also None.
    y_data : pandas.DataFrame or None
        Typical load profiles of the training points.
        If None, default values are used.
    regressor : object
        Regression model object with `fit` and `predict` methods.

    Attributes
    ----------
    x_data : pandas.DataFrame or None
        ToU energy consumption values of the training points.
        If None, default values are used.
    y_data : pandas.DataFrame or None
        Typical load profiles of the training points.
        If None, default values are used.
    regressor : object or None
        Regression model object with `fit` and `predict` methods.
        If None, KNeighborsRegressor from sklearn is used.

    Methods
    -------
    predict(x)
        Predict typical load profiles based on the provided inputs.
    add_data_points(x, y)
        Add new data points to the predictor.
    """

    def __init__(self, x_data=None, y_data=None, regressor=None,):

        if regressor is not None:
            assert hasattr(regressor, 'fit') and \
                   hasattr(regressor, 'predict'), \
                "'regressor' must have 'fit' and 'predict' methods."
        else:
            regressor = default_regressor
        self.regressor = regressor

        assert (x_data is None) == (y_data is None), \
            "'x_data' and 'y_data' must both be None or not be None."
        if y_data is None:
            y_data = def_y_data
            x_data = def_x_data
        super().__init__(x_data=x_data, y_data=y_data)

    def predict(self, x, nd=None, scaler=None):
        """
        Predict typical load profiles based on the provided inputs.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values of the point(s) to predict.
        nd : array-like or None, optional
            Number of days of each day-type for each point.
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
            If any inconsistency is found in the input data.
        """
        # Check consistency of input data
        x, nd, n_points, mono_dim = self.check_input_data(x, nd)

        # If scaling is required, nd cannot be None as needed for scaling
        if scaler is not None:
            assert nd is not None, \
                "'nd' cannot be None, since needed for scaling."

        # Transform 'nd' to list of None for compatibility with zip
        nd = [None] * n_points if nd is None else nd

        # Predict typical load profile of all points
        y_pred = self.regressor.predict(x)

        if scaler is not None:
            # Scale each predicted profile individually
            for i in range(n_points):
                y_pred[i] = scaler(y=y_pred[i], x_des=x[i], nd=nd[i])

        if mono_dim:
            return y_pred[0]
        else:
            return y_pred

    def _fit(self):
        """
        Train the regression model.

        Parameters
        ----------

        """
        self.regressor.fit(self.x_data.values, self.y_data.values)

    def add_data_points(self, x_data, y_data, **kwargs):
        """
        Add new training points to the predictor and re-train the regressor.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values of the new data points.
        y : array-like
            Typical load profiles of the new data points.

        Additional parameters
        ---------------------
        Parameters from BasePredictor.add_data_points method.

        Raises
        ------
        AssertionError
            If 'x_data' and 'y_data' do not comply with requirements for new
            data points.
        """

        # Call the base class method to add data points
        super().add_data_points(x_data, y_data, **kwargs)

        # Re-train the regression model using the updated data
        self._fit()


# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from bill2watt.scaling import flat

    # Example usage
    regression_predictor = RegressionPredictor()

    # Example usage with one point
    x = np.array([1, 2, 3])
    nd = np.array([22, 4, 5])
    y = regression_predictor.predict(x, nd)
    print("1 point evaluated, output shape: ", y.shape)
    plt.plot(y)
    plt.show()

    # Example usage with two points and scaling
    x = np.array([[1, 2, 3], [4, 5, 6]])
    nd = np.array([[22, 4, 5], [22, 4, 5]])
    y = regression_predictor.predict(x, nd, scaler=flat.evaluate)
    print("2 points evaluated, output shape: ", y.shape)
    plt.plot(y.T)
    plt.show()
