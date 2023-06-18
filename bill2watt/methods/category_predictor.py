"""
This module contains the 'CategoryPredictor' class, which is used for
predicting typical load profiles based on monthly time-of-use (ToU) energy
consumption values, using training data and an a-priori categorization
approach.

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

# Path for the folder of common data
basepath = path.dirname(path.abspath(__file__))
folder = path.join(basepath, "data")

# Default training dataset
def_c_data = pd.read_csv(path.join(folder, "c_train.csv"), sep=';',
                         index_col=[0, 1])['class'].to_frame()
def_y_data = def_c_data.join(pd.read_csv(path.join(folder, "y_train_norm.csv"),
                                         sep=';', index_col=[0, 1])). \
    set_index(list(def_c_data.columns), append=True)
def_x_data = def_c_data.join(pd.read_csv(path.join(folder, "x_train_norm.csv"),
                                         sep=';', index_col=[0, 1])). \
    set_index(list(def_c_data.columns), append=True)


class CategoryPredictor(BasePredictor):
    """
    This class implements a predictor for evaluating typical load profiles
    based on monthly time-of-use (ToU) energy consumption values, using
    categories defined by a specific key.

    Parameters
    ----------
    x_data : pandas.DataFrame or None
        ToU energy consumption values associated with the training points.
        If None, default values are used.
    y_data : pandas.DataFrame or None
        Typical load profiles of the training points.
        If None, default values are used.
    category : str or None, optional
        The name of the column/index level used for categorization.
        If None and 'y_data' is not None, the name of the index of 'y_data'
        is used.

    Attributes
    ----------
    x_data (property) : pandas.DataFrame or None
        ToU energy consumption values of the training points.
        If None, default values are used.
    y_data (property) : pandas.DataFrame or None
        Typical load profiles of the training points.
        If None, default values are used.
    category (property) : str or None
        The name of the column/index level used for categorization.
    y_categories (property) : pandas.DataFrame or None
        Categorized typical load profiles based on the training data.
        If None, no categorization has been performed yet.
    categories (property) : list or None
        Unique category values present in the categorized data.
        If None, no categorization has been performed yet.

    Methods
    -------
    predict(x, key)
        Evaluate typical load profiles based on ToU energy consumption values
        and using the categories.
    add_data(x_data, y_data, **kwargs)
        Add new data points to the predictor and re-train the categorizer.

    See Also
    --------
    BasePredictor : Base class for typical load profile predictors.
    """

    def __init__(self, x_data=None, y_data=None, category=None):
        """
        Initialize the CategoryPredictor.

        Parameters
        ----------
        x_data : pandas.DataFrame or None, optional
            ToU energy consumption values associated with the training points.
            If None, default values are used.
        y_data : pandas.DataFrame or None, optional
            Typical load profiles of the training points.
            If None, default values are used.
        category : str or None, optional
            The name of the column/index level used for categorization.
            If None and 'y_data' is not None, the name of the index of 'y_data'
            is used.
        """
        assert (x_data is None) == (y_data is None), \
            "'x_data' and 'y_data' must both be None or not be None."
        if y_data is None:
            y_data = def_y_data
            x_data = def_x_data
            category = 'class' if category is None else category

        self._category = category
        self._y_categories = None
        self._category = self._validate_category(category, y_data)

        super().__init__(x_data=x_data, y_data=y_data)

    @property
    def category(self):
        """
        Get or set the name of the column/index level used for categorization.

        Returns
        -------
        str or None
            The name of the column/index level used for categorization.
            If None, no categorization has been performed yet.
        """
        return self._category

    @category.setter
    def category(self, category):
        """
        Set the name of the column/index level used for categorization.

        Parameters
        ----------
        category : str or None
            The name of the column/index level used for categorization.
            If None and 'y_data' is not None, the name of the index of 'y_data'
            is used.

        Raises
        ------
        AssertionError
            If 'category' is not valid. See '_validate_category' method for
            further details.
        """
        self._category = self._validate_category(category, self.y_data)
        self._fit()

    @property
    def y_categories(self):
        """
        Categorized typical load profiles based on the training data.

        Returns
        -------
        pandas.DataFrame or None
            Categorized typical load profiles.
            If None, no categorization has been performed yet.
        """
        if self._y_categories is None:
            return None
        return self._y_categories.copy()

    @property
    def categories(self):
        """
        Unique category values present in the categorized data.

        Returns
        -------
        list or None
            Unique category values.
            If None, no categorization has been performed yet.
        """
        if self.y_categories is None:
            return None
        return list(self._y_categories.index())

    @staticmethod
    def _validate_category(category, y_data):
        """
        Validate the category used for categorization.

        If 'category' is None, the name of the index level of 'y_data' is used.

        Parameters
        ----------
        category : str or None
            The name of the column/index level used for categorization.
            If None, the name of the index of 'y_data' is used.

        Returns
        -------
        str
            The validated category name.

        Raises
        ------
        AssertionError
            - If 'category' is not present in the index of 'y_data' when
              provided as an argument.
            - If more categorization levels are found and 'category' is None.
        """
        if category is None:
            assert y_data.index.nlevels == 1, \
                "Multiple categorization levels found but 'category' is None."
            category = y_data.index.names[0]
        else:
            assert category in y_data.index.names, \
                "'category' is not present in the index of 'y_data'."
        return category

    def _fit(self):
        """
        Perform categorization based on the training data.

        Raises
        ------
        AssertionError
            - If 'category' is not present in the index of 'y_data' when
              provided as an argument.
        """
        self._y_categories = self.y_data.groupby(level=self.category).mean()

    def predict(self, x, key):
        """
        Evaluate typical load profiles based on ToU energy consumption values
        and using the categories.

        Parameters
        ----------
        x : array-like
            ToU energy consumption values of the point(s) to predict.
        key : tuple or list of tuples
            Key values related to the categorized data associated with each
            point.

        Returns
        -------
        np.ndarray
            Predicted typical load profiles based on the provided inputs.

        Raises
        ------
        AssertionError
            If any inconsistency is found in the input data or key values are
            not present in the categorized data.
        """
        # Check consistency of input data
        x, n_points, mono_dim = self._validate_input(x)

        # Check that the right number of correct keys is provided
        key = (key,) if n_points == 1 and mono_dim else key
        assert len(key) == n_points, \
            "Length of 'key' must be equal to {}.".format(n_points)

        # Initialize a list of "predicted" typical load profiles
        y_pred = []

        # "Predict" typical load profile of each point using the categorized data
        for x_, key_ in zip(x, key):
            assert key_ in self.y_categories.index, \
                "Key {} is not present in the categories index.".format(key_)

            y_pred.append(self.y_categories.loc[key_, :].values)

        if mono_dim:
            return np.array(y_pred)[0]
        else:
            return np.array(y_pred)

    def add_data(self, x_data, y_data, **kwargs):
        """
        Add new training points to the predictor and re-train the categorizer.

        Parameters
        ----------
        x_data : array-like
            ToU energy consumption values of the new data points.
        y_data : array-like
            Typical load profiles of the new data points.

        Additional Parameters
        ---------------------
        Parameters from the 'add_data' method of the 'BasePredictor' parent
        class.

        Raises
        ------
        AssertionError
            If 'x_data' and 'y_data' do not comply with the requirements for new
            data points.
        """
        # Call the base class method to add data points
        super().add_data(x_data, y_data, **kwargs)

        self._fit()

# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example usage
    predictor = CategoryPredictor()

    # Example usage with one point
    x = np.array([1, 2, 3])
    key = ('bta1')
    y = predictor.predict(x, key)
    print("1 point evaluated, output shape: ", y.shape)
    plt.plot(y)
    plt.show()

    # Example usage with two points and scaling
    x = np.array([[1, 2, 3], [4, 5, 6]])
    key = (('bta1'), ('bta4'))
    y = predictor.predict(x, key)
    print("2 points evaluated, output shape: ", y.shape)
    plt.plot(y.T)
    plt.show()