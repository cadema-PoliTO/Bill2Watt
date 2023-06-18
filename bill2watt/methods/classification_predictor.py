"""
This module contains the 'ClassificationPredictor' class, which is used for
predicting typical load profiles based on monthly time-of-use (ToU) energy
consumption values, using training data and a classification approach.

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
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from bill2watt.methods.base_predictor import BasePredictor

# Path for the folder of common data
basepath = path.dirname(path.abspath(__file__))
folder = path.join(basepath, "data")

# Default training dataset
def_y_data = pd.read_csv(path.join(folder, "y_train_norm.csv"), sep=';',
                         index_col=[0, 1])
def_x_data = pd.read_csv(path.join(folder, "x_train_norm.csv"), sep=';',
                         index_col=[0, 1])

# Default clustering model
def_clustering = KMeans(n_clusters=11, n_init=100)

# Default classification model
def_classifier = DecisionTreeClassifier(max_depth=5)


class ClassificationPredictor(BasePredictor):
    """
    This class implements a predictor for evaluating typical load profiles
    based on time-of-use (ToU) energy consumption values using training
    data in a classification and clustering approach.

    Parameters
    ----------
    x_data : pandas.DataFrame or None
        ToU energy consumption values of the training points.
        If None, default values are used.
    y_data : pandas.DataFrame or None
        Typical load profiles of the training points.
        If None, default values are used.
    clustering : object
        Clustering model object with `fit` and `predict` methods.
    classifier : object
        Classification model object with `fit` and `predict` methods.

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

    def __init__(self, x_data=None, y_data=None, clustering=None, 
                 classifier=None):

        if clustering is not None:
            assert hasattr(clustering, 'fit') and \
                   hasattr(clustering, 'labels_') and \
                   hasattr(clustering, 'cluster_centers_'), \
            "'clustering' must have 'fit' method and 'labels_' and " \
            "'cluster_centers_ attributes."
        else:
            clustering = def_clustering
        self._clustering = clustering

        if classifier is not None:
            assert hasattr(classifier, 'fit') and \
                   hasattr(classifier, 'predict'), \
                "'classifier' must have 'fit' and 'predict' methods."
        else:
            classifier = def_classifier
        self._classifier = classifier

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
        If 'x' refers to a single point, then the output is provided as a
        single point. If 'x' refers to multiple points, the output is provided
        as a matrix with the same number of rows as 'x'.

        Raises
        ------
        AssertionError
            If any inconsistency is found in the input data.
        """
        # Check consistency of input data
        x, n_points, mono_dim = self._validate_input(x)

        # Predict typical load profile
        # Cluster labels
        labels = self._classifier.predict(x)
        # Get cluster centers
        y_pred = self._clustering.cluster_centers_[labels]

        if mono_dim:
            return y_pred[0]
        else:
            return y_pred

    def _fit(self):
        """
        Train the clustering and classification models.

        Parameters
        ----------

        """
        # Fit the clustering model
        self._clustering.fit(self.y_data.values)
        print('cacaz, ', sorted([np.count_nonzero(self._clustering.labels_ == l)
                                 for l in set(self._clustering.labels_)]))
        print('cacaz, ', self._clustering.inertia_)

        # Assign cluster labels to training data
        cluster_labels = self._clustering.predict(self.y_data.values)

        # Train the classification model with cluster labels as target
        self._classifier.fit(self.x_data.values, cluster_labels.reshape(-1, 1))

    def add_data(self, x_data, y_data, **kwargs):
        """
        Add new training points to the predictor and re-train the models.

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

        # Re-train the models using the updated data
        self._fit()

# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example usage
    predictor = ClassificationPredictor()

    # Example usage with one point
    x = np.array([100, 20, 300])
    y = predictor.predict(x)
    print("1 point evaluated, output shape: ", y.shape)
    plt.plot(y)
    plt.show()

    # Example usage with two points
    x = np.array([[100, 20, 300], [500, 40, 50]])
    y = predictor.predict(x)
    print("2 points evaluated, output shape: ", y.shape)
    plt.plot(y.T)
    plt.show()
