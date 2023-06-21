"""
This module contains the 'RegressionPredictor' class, which is used for
predicting typical load profiles based on monthly time-of-use (ToU) energy
consumption values, using training data and a regression approach.

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
from bill2watt.predictors.base_predictor import BasePredictor

# Path for the folder of common data
basepath = path.dirname(path.abspath(__file__))
folder = path.join(basepath, "data")

# Default training dataset
def_y_data = pd.read_csv(path.join(folder, "y_train_norm.csv"), sep=';',
                         index_col=[0, 1])
def_x_data = pd.read_csv(path.join(folder, "x_train_norm.csv"), sep=';',
                         index_col=[0, 1])

# Default regressor
def_regressor = KNeighborsRegressor(n_neighbors=9)


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
    x_data (property) : pandas.DataFrame or None
        ToU energy consumption values of the training points.
        If None, default values are used.
    y_data (property) : pandas.DataFrame or None
        Typical load profiles of the training points.
        If None, default values are used.

    Methods
    -------
    predict(x)
        Predict typical load profiles based on the provided inputs.
    add_data(x, y)
        Add new data points to the predictor.

    See Also
    --------
    BasePredictor : Base class for typical load profile predictors.
    """

    def __init__(self, x_data=None, y_data=None, regressor=None,):

        regressor = def_regressor if regressor is None else regressor
        assert hasattr(regressor, 'fit') and hasattr(regressor, 'predict'),\
            "'regressor' must have 'fit' and 'predict' methods."
        self._regressor = regressor

        assert (x_data is None) == (y_data is None), \
            "'x_data' and 'y_data' must both be None or not be None."
        if y_data is None:
            y_data = def_y_data
            x_data = def_x_data
        super().__init__(x_data=x_data, y_data=y_data)

    def predict(self, x):
        """
        Predict typical load profiles based on the provided inputs.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values of the point(s) to predict.

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
        x, n_points, mono_dim = self._validate_input(x)

        # Predict typical load profile of all points
        y_pred = self._regressor.predict(x)

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
        self._regressor.fit(self.x_data.values, self.y_data.values)

    def add_data(self, x_data, y_data, **kwargs):
        """
        Add new training points to the predictor and re-train the regressor.

        Parameters
        ----------
        x_data : array-like
            ToU energy consumption values of the new data points.
        y_data : array-like
            Typical load profiles of the new data points.

        Additional parameters
        ---------------------
        Parameters from 'add_data' method of 'BasePredictor' parent class.

        Raises
        ------
        AssertionError
            If 'x_data' and 'y_data' do not comply with requirements for new
            data points.
        """

        # Call the base class method to add data points
        super().add_data(x_data, y_data, **kwargs)

        # Re-train the regression model using the updated data
        self._fit()


# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example usage
    regression_predictor = RegressionPredictor()

    # Example usage with one point
    x = np.array([1, 2, 3])
    x = x / x.sum()
    y = regression_predictor.predict(x)
    print("1 point evaluated, output shape: ", y.shape)
    plt.plot(y)
    plt.show()

    # Example usage with two points
    x = np.array([[1, 2, 3], [4, 5, 6]])
    x = x / x.sum(axis=1)[:, np.newaxis]
    y = regression_predictor.predict(x)
    print("2 points evaluated, output shape: ", y.shape)
    plt.plot(y.T)
    plt.show()
