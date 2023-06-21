"""

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import numpy as np


def check_array(x):
    """
    Validate the input array and return a reshaped version if necessary.

    Parameters
    ----------
    x : ndarray
        The input array to validate.

    Returns
    -------
    ndarray
        The reshaped input array.
    bool
        Whether the input array is mono-dimensional or not.

    Raises
    ------
    AssertionError
        If the input is not a 1D or 2D NumPy array.
    """
    assert isinstance(x, np.ndarray), "Input 'x' must be a NumPy array."
    assert x.ndim == 1 or x.ndim == 2, "Input 'x' must be 1D or 2D."
    return x[:, np.newaxis] if x.ndim == 1 else x, x.ndim == 1


class XRowNormalizer:
    """
    Row-wise normalizer for X data.

    This class provides methods to normalize and denormalize row-wise data
    based on the sum of each row.

    Parameters
    ----------

    Attributes
    ----------
    x_sum (property) : ndarray or None
        Normalization values in the X data.
    n (property) : int
        Number of points in the fitting data.

    Methods
    -------
    fit(x)
        Fit the normalizer to the given X data.
    transform(x)
        Normalize the input X data.
    inverse_transform(x_norm)
        Denormalize the normalized X data.
    fit_transform(x)
        Fit the normalizer to the given X data and normalize it.

    """

    def __init__(self):
        self._x_sum = None

    def __bool__(self):
        return self.x_sum is not None

    @property
    def x_sum(self):
        """
        Get the normalization values of the X data.

        Returns
        -------
        ndarray or None
            Values of self._x_sum

        """
        return self._x_sum

    @property
    def n(self):
        """
        Get the number of rows in the X data.

        Returns
        -------
        int or None
            The number of rows in the X data.
        """
        return self._x_sum.size if self else None

    def fit(self, x):
        """
        Fit the normalizer to the given X data.

        Parameters
        ----------
        x : ndarray
            The input X data.

        Raises
        ------
        AssertionError
            If the input 'x' does not meet the requirements. See function
            'check_array' for further details.
        """
        x, monodim = check_array(x)

        x_sum = x.sum(axis=1)[:, np.newaxis]
        assert np.all(x_sum != 0), "Input 'x' contains rows with zero sum."

        self._x_sum = x_sum

    def transform(self, x):
        """
        Normalize the input X data.

        Parameters
        ----------
        x : ndarray
            The input X data.

        Returns
        -------
        ndarray
            The normalized X data.

        Raises
        ------
        AssertionError
            If the input 'x' does not meet the requirements. See function
            'check_array' for further details. If the length of 'x' does not
            match the number of rows in the X data.
        """
        assert self, "Normalizer is not fitted."
        x, monodim = check_array(x)
        assert len(x) == self.n, \
            "Input 'x' must have length {}.".format(self.n)

        x_norm = x / self._x_sum

        return x_norm.flatten() if monodim else x_norm

    def inverse_transform(self, x_norm):
        """
        Denormalize the normalized X data.

        Parameters
        ----------
        x_norm : ndarray
            The normalized X data.

        Returns
        -------
        ndarray
            The denormalized X data.

        Raises
        ------
        AssertionError
            If the input 'x_norm' does not meet the requirements. See function
            'check_array' for further details. If the length of 'x' does not
            match the number of rows in the X data.
        """
        assert self, "Normalizer is not fitted."
        x_norm, monodim = check_array(x_norm)
        assert len(x_norm) == self.n, \
            "Input 'x_norm' must have length {}.".format(self.n)

        x = x_norm * self._x_sum
        return x.flatten() if monodim else x

    def fit_transform(self, x):
        """
        Fit the normalizer to the given X data and normalize it.

        Parameters
        ----------
        x : ndarray
            The input X data.

        Returns
        -------
        ndarray
            The normalized X data.

        Notes
        -----
        See 'fit' and 'transform' methods for further details.
        """
        self.fit(x)
        return self.transform(x)


class YRowNormalizer(XRowNormalizer):
    """
    Row-wise normalizer for Y data.

    This class provides methods to normalize and denormalize row-wise data
    based on the sum of each row, using the sum of X data for normalization.

    Attributes
    ----------
    x_sum (property) : ndarray or None
        Normalization values in the X data.
    n (property) : int
        Number of points in the fitting data.

    Methods
    -------
    transform(y)
        Normalize the input Y data.
    inverse_transform(y_norm)
        Denormalize the normalized Y data.
    fit_transform(y)
        Fit the normalizer to the given Y data and normalize it.
    """

    def transform(self, y):
        """
        Normalize the input Y data.

        Parameters
        ----------
        y : ndarray
            The input Y data.

        Returns
        -------
        ndarray
            The normalized Y data.

        Raises
        ------
        AssertionError
            If the input 'y' does not meet the requirements. See function
            'check_array' for further details. If the length of 'y' does not
            match the number of rows in the Y data.
        """
        "Normalizer is not fitted."
        y, monodim = check_array(y)
        assert len(y) == self.n, \
            "Input 'y' must have length {}.".format(self.n)

        y_norm = y / self._x_sum

        return y_norm.flatten() if monodim else y_norm

    def inverse_transform(self, y_norm):
        """
        Denormalize the normalized Y data.

        Parameters
        ----------
        y_norm : ndarray
            The normalized Y data.

        Returns
        -------
        ndarray
            The denormalized Y data.

        Raises
        ------
        AssertionError
            If the input 'y_norm' does not meet the requirements. See function
            'check_array' for further details. If the length of 'y' does not
            match the number of rows in the Y data.
        """
        "Normalizer is not fitted."
        y_norm, monodim = check_array(y_norm)
        assert len(y_norm) == self.n, \
            "Input 'y_norm' must have length {}.".format(self.n)

        y = y_norm * self._x_sum
        return y.flatten() if monodim else y

    def fit_transform(self, x, y):
        """
        Fit the normalizer to the given X data and normalize the Y data.

        Parameters
        ----------
        x : ndarray
            The input X data.
        y : ndarray
            The input Y data.

        Returns
        -------
        ndarray
            The normalized Y data.

        Notes
        -----
        See 'fit' and 'transform' methods for further details.
        """
        self.fit(x)
        return self.transform(y)


# -------------------------------------------------------------------- EXAMPLE
if __name__ == "__main__":
    # Example data
    n = 3
    x_data = np.array(np.random.rand(n))
    y_data = np.array(np.random.rand(n, 3))

    # Create an instance of XRowNormalizer
    x_normalizer = XRowNormalizer()
    # Fit the normalizer to X data
    x_normalizer.fit(x_data)
    # Transform X data
    x_normalized = x_normalizer.transform(x_data)
    # Inverse transform X data
    x_denormalized = x_normalizer.inverse_transform(x_normalized)
    # Print the results
    print("Original X data:")
    print(x_data)
    print("Normalized X data:")
    print(x_normalized)
    print("Denormalized X data:")
    print(x_denormalized)

    # Create an instance of YRowNormalizer
    y_normalizer = YRowNormalizer()
    # Fit the normalizer to X and Y data and transform Y data
    y_normalized = y_normalizer.fit_transform(x_data, y_data)
    # Inverse transform Y data
    y_denormalized = y_normalizer.inverse_transform(y_normalized)

    # Print the results
    print("Original Y data:")
    print(y_data)
    print("Normalized Y data:")
    print(y_normalized)
    print("Denormalized Y data:")
    print(y_denormalized)